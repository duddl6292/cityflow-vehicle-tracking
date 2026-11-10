"""CityFlowV2의 주석 차량을 대상으로 두 추적 결과를 비교한다.

평가 대상은 입력 GT의 박스 행이다. 프레임별 일대일 IoU 매칭으로
박스 매칭률, ID 변경, 미연결 후 재연결 구간을 집계한다.
"""

import json
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_gt(path):
    """GT의 left, top, width, height를 xyxy 좌표로 변환한다."""
    frames = defaultdict(list)

    with path.open(encoding="utf-8-sig", newline="") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if not row:
                continue

            try:
                frame_id, gt_id = map(int, row[:2])
                left, top, width, height = map(float, row[2:6])
            except ValueError as error:
                raise ValueError(f"{path}의 {line_number}번째 GT 행 형식을 확인하세요.") from error

            box = (left, top, left + width, top + height)
            frames[frame_id].append((gt_id, box))

    if not frames:
        raise ValueError(f"GT 박스가 없습니다. {path}")

    validate_boxes(frames, path)
    return frames


def load_tracks(path):
    """추적 CSV를 프레임 번호별로 묶는다."""
    frames = defaultdict(list)
    coordinate_columns = ("x1", "y1", "x2", "y2")
    required_columns = {"frame_id", "track_id", *coordinate_columns}

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{path}에 필요한 열이 없습니다. {', '.join(sorted(missing_columns))}")

        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                track_id = int(row["track_id"])
                box = tuple(float(row[column]) for column in coordinate_columns)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}의 {reader.line_num}번째 추적 행 형식을 확인하세요.") from error

            frames[frame_id].append((track_id, box))

    validate_boxes(frames, path)
    return frames


def validate_boxes(frames, path):
    """중복 ID와 잘못된 좌표가 집계에 섞이지 않도록 확인한다."""
    for frame_id, boxes in frames.items():
        if frame_id < 1:
            raise ValueError(f"{path}의 프레임 번호는 1 이상이어야 합니다.")

        ids = set()
        for identity, box in boxes:
            if identity in ids:
                raise ValueError(f"{path}의 프레임 {frame_id}에 ID {identity}가 중복됩니다.")
            ids.add(identity)

            x1, y1, x2, y2 = box
            if not np.isfinite(box).all() or x2 <= x1 or y2 <= y1:
                raise ValueError(f"{path}의 프레임 {frame_id}, ID {identity} 박스가 잘못됐습니다.")


def match_frame(gt_boxes, track_boxes, min_iou):
    """한 프레임에서 GT ID와 추적 ID를 일대일로 연결한다."""
    if not track_boxes:
        return {}

    gt_coords = np.array([box for _, box in gt_boxes])
    track_coords = np.array([box for _, box in track_boxes])

    # 행은 GT, 열은 추적 박스다. 모든 박스 쌍의 교집합을 계산한다.
    top_left = np.maximum(gt_coords[:, None, :2], track_coords[None, :, :2])
    bottom_right = np.minimum(gt_coords[:, None, 2:], track_coords[None, :, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=2)

    gt_areas = (gt_coords[:, 2:] - gt_coords[:, :2]).prod(axis=1)
    track_areas = (track_coords[:, 2:] - track_coords[:, :2]).prod(axis=1)
    union = gt_areas[:, None] + track_areas[None, :] - intersection
    ious = intersection / union

    # 기준을 통과한 연결 수를 먼저 최대화하고, 그 안에서 IoU 합을 높인다.
    valid = ious >= min_iou
    match_bonus = min(ious.shape) + 1
    scores = np.where(valid, match_bonus + ious, 0)
    gt_indices, track_indices = linear_sum_assignment(scores, maximize=True)

    return {
        gt_boxes[gt_index][0]: track_boxes[track_index][0]
        for gt_index, track_index in zip(gt_indices, track_indices)
        if valid[gt_index, track_index]
    }


def evaluate(gt_frames, track_frames, min_iou):
    # 각 GT 차량에 대해 관측 프레임과 대응하는 추적 ID를 시간순으로 모은다.
    histories = defaultdict(list)
    for frame_id in sorted(gt_frames):
        gt_boxes = gt_frames[frame_id]
        matches = match_frame(gt_boxes, track_frames.get(frame_id, []), min_iou)
        for gt_id, _ in gt_boxes:
            histories[gt_id].append((frame_id, matches.get(gt_id)))

    matched_boxes = 0
    id_changes = 0
    reconnections = 0

    for history in histories.values():
        previous_gt_frame = None
        last_matched_frame = None
        last_track_id = None

        for frame_id, track_id in history:
            # GT 주석이 건너뛴 구간은 새 구간으로 평가한다.
            if previous_gt_frame is None or frame_id != previous_gt_frame + 1:
                last_matched_frame = None
                last_track_id = None
            previous_gt_frame = frame_id

            if track_id is None:
                continue

            matched_boxes += 1
            if last_track_id is not None and track_id != last_track_id:
                id_changes += 1
            if last_matched_frame is not None and frame_id > last_matched_frame + 1:
                reconnections += 1

            last_matched_frame = frame_id
            last_track_id = track_id

    return (matched_boxes, id_changes, reconnections), histories


def main():
    parser = argparse.ArgumentParser(description="GT 박스를 기준으로 두 추적 결과를 비교합니다.")
    parser.add_argument("--gt", 
                        required=True, 
                        type=Path, 
                        help="카메라의 gt.txt 경로")
    parser.add_argument("--high-only", 
                        required=True, 
                        type=Path, 
                        help="1차 매칭 결과 CSV 경로")
    parser.add_argument("--two-stage", 
                        required=True, 
                        type=Path, 
                        help="2차 매칭 결과 CSV 경로")
    parser.add_argument("--min-iou", 
                        type=float, 
                        default=0.5, 
                        help="평가용 IoU 하한 (기본값 0.5)")
    parser.add_argument("--output-dir", 
                        type=Path, 
                        help="평가 결과를 JSON으로 저장할 폴더")
    args = parser.parse_args()

    if not 0 < args.min_iou <= 1:
        parser.error("--min-iou는 0보다 크고 1 이하여야 합니다.")

    try:
        gt_frames = load_gt(args.gt)
        high, high_histories = evaluate(
            gt_frames, load_tracks(args.high_only), args.min_iou
        )
        two, two_histories = evaluate(
            gt_frames, load_tracks(args.two_stage), args.min_iou
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    gt_count = sum(len(boxes) for boxes in gt_frames.values())
    vehicle_count = len({gt_id for boxes in gt_frames.values() for gt_id, _ in boxes})
    high_rate = high[0] / gt_count * 100
    two_rate = two[0] / gt_count * 100

    print(f"GT 차량 {vehicle_count}대, 박스 {gt_count:,}개, 평가 IoU {args.min_iou:g}")
    print("\n| 지표 | 1차 매칭만 | 2차 매칭 추가 |")
    print("| --- | ---: | ---: |")
    print(f"| 연결된 GT 박스 | {high[0]:,} | {two[0]:,} |")
    print(f"| GT 박스 매칭률 | {high_rate:.2f}% | {two_rate:.2f}% |")
    print(f"| ID 변경 | {high[1]} | {two[1]} |")
    print(f"| 미연결 후 재연결 구간 | {high[2]} | {two[2]} |")
    print(f"\nGT 박스 매칭률 변화 {two_rate - high_rate:+.2f}%p")
    if args.output_dir is not None:
        metrics = {
            "evaluation_min_iou": args.min_iou,
            "gt_vehicle_count": vehicle_count,
            "gt_box_count": gt_count,
        }

        # evaluate()가 반환한 세 값을 풀어서 의미가 드러나는 이름으로 저장합니다.
        for name, result, rate in (
            ("high_only", high, high_rate),
            ("two_stage", two, two_rate),
        ):
            matched_boxes, id_changes, reconnections = result

            metrics[name] = {
                "matched_gt_boxes": matched_boxes,
                "gt_match_rate_percent": round(rate, 2),
                "id_changes": id_changes,
                "reconnections": reconnections,
            }

        # 상위 폴더까지 만들고, 폴더가 이미 있어도 그대로 사용합니다.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "metrics.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        print(f"\n평가 결과 저장: {output_path.resolve()}")
        # 2차 결과에서 (프레임 번호, GT 차량 ID)로 추적 ID를 찾을 수 있게 합니다.
        two_by_observation = {
            (frame_id, gt_id): track_id
            for gt_id, history in two_histories.items()
            for frame_id, track_id in history
        }

        events_path = args.output_dir / "case_events.jsonl"
        with events_path.open("w", encoding="utf-8") as file:
            for gt_id, history in high_histories.items():
                for frame_id, high_id in history:
                    two_id = two_by_observation[(frame_id, gt_id)]

                    # 한 줄이 특정 프레임에서 관측된 GT 차량 한 대의 비교 기록입니다.
                    event = {
                        "frame_id": frame_id,
                        "gt_id": gt_id,
                        "high_only_track_id": high_id,
                        "two_stage_track_id": two_id,
                    }
                    file.write(json.dumps(event) + "\n")

        print(f"프레임별 비교 저장: {events_path.resolve()}")
if __name__ == "__main__":
    main()
