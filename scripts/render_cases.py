"""CityFlow 정답 차량 한 대의 추적 결과를 프레임별 이미지로 비교한다."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from evaluate_tracking import load_gt, load_tracks, match_frame
from inspect_detections import calculate_iou


def describe_track(gt_id, gt_boxes, track_boxes, min_iou):
    # 다른 GT도 함께 평가해야 기존 일대일 매칭과 같은 결과를 얻는다.
    gt_box = dict(gt_boxes)[gt_id]
    matched_id = match_frame(gt_boxes, track_boxes, min_iou).get(gt_id)
    if matched_id is not None:
        selected_id, selected_box = matched_id, dict(track_boxes)[matched_id]
        status = "matched"
    else:
        selected_id, selected_box = max(
            track_boxes, key=lambda item: calculate_iou(gt_box, item[1]),
            default=(None, None),
        )
        if selected_box is None or calculate_iou(gt_box, selected_box) <= 0:
            return {"status": "no_overlap", "track_id": None, "iou": None, "box": None}
        status = "candidate"

    return {
        "status": status,
        "track_id": selected_id,
        "iou": calculate_iou(gt_box, selected_box),
        "box": selected_box,
    }


def make_panel(frame, crop, gt_box, title, subtitle, result=None):
    left, top, right, bottom = crop
    view = frame[top:bottom, left:right]
    scale = min(620 / view.shape[1], 360 / view.shape[0])
    width = max(1, round(view.shape[1] * scale))
    height = max(1, round(view.shape[0] * scale))
    view = cv2.resize(view, (width, height))
    x_offset, y_offset = (640 - width) // 2, 76 + (360 - height) // 2
    panel = np.full((446, 640, 3), 24, dtype=np.uint8)
    panel[y_offset:y_offset + height, x_offset:x_offset + width] = view

    def draw_box(box, color, thickness):
        x1, y1, x2, y2 = box
        point1 = (x_offset + round((x1 - left) * width / (right - left)),
                  y_offset + round((y1 - top) * height / (bottom - top)))
        point2 = (x_offset + round((x2 - left) * width / (right - left)),
                  y_offset + round((y2 - top) * height / (bottom - top)))
        cv2.rectangle(panel, point1, point2, color, thickness)

    draw_box(gt_box, (90, 220, 80), 4)
    if result and result["box"] is not None:
        color = (255, 200, 70) if result["status"] == "matched" else (30, 150, 255)
        draw_box(result["box"], color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, title, (15, 30), font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (15, 59), font, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def comparison_image(frame, frame_id, gt_id, gt_box, high, two, min_iou):
    # 세 패널에서 동일 영역을 확대해 박스 위치 차이를 비교한다.
    boxes = [gt_box] + [result["box"] for result in (high, two) if result["box"]]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    frame_height, frame_width = frame.shape[:2]
    padding = max(28, round(max(right - left, bottom - top) * 0.25))
    crop = (max(0, int(left) - padding), max(0, int(top) - padding),
            min(frame_width, round(right) + padding), min(frame_height, round(bottom) + padding))
    if crop[0] >= crop[2] or crop[1] >= crop[3]:
        raise ValueError(f"{frame_id}번 프레임에서 GT와 영상 좌표가 맞는지 확인하세요.")

    gt_panel = make_panel(frame, crop, gt_box, f"Frame {frame_id} | GT {gt_id}",
                          f"Green: ground truth | Eval min IoU: {min_iou:g}")
    panels = [gt_panel]
    for title, result in (("High only", high), ("Two stage", two)):
        if result["status"] == "no_overlap":
            subtitle = "Unmatched | No overlapping track box"
        else:
            status = "Matched" if result["status"] == "matched" else "Unmatched | Best candidate"
            subtitle = f"{status} #{result['track_id']} | IoU {result['iou']:.3f}"
        panels.append(make_panel(frame, crop, gt_box, title, subtitle, result))
    return np.hstack(panels)


def save_case(path, case):
    # 같은 실행·차량·구간을 다시 저장하면 해당 사례 한 줄을 갱신한다.
    records = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                records.append(json.loads(line))
    key = (case["run_id"], case["case_id"])
    records = [row for row in records if (row.get("run_id"), row.get("case_id")) != key]
    records.append(case)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="두 추적 CSV가 있는 탐지 실행 폴더")
    parser.add_argument("--gt-id", type=int, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, help="생략하면 시작 프레임 한 장만 저장")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.5, help="기존 평가에 사용한 IoU 기준")
    parser.add_argument("--observation", help="이미지를 확인한 뒤 작성한 메모. 입력하면 사례도 저장")
    args = parser.parse_args()
    end_frame = args.start_frame if args.end_frame is None else args.end_frame
    if not 1 <= args.start_frame <= end_frame or not 0 < args.min_iou <= 1:
        parser.error("프레임 범위 또는 평가 IoU 값을 확인하세요.")

    gt = load_gt(args.camera_dir / "gt" / "gt.txt")
    high_tracks = load_tracks(args.run_dir / "tracks_high_only.csv")
    two_tracks = load_tracks(args.run_dir / "tracks_two_stage.csv")
    selected = [f for f in range(args.start_frame, end_frame + 1) if args.gt_id in dict(gt.get(f, []))]
    if not selected:
        parser.error("지정한 구간에 이 차량의 GT 주석이 없습니다.")

    run_id = args.run_dir.resolve().name
    camera = f"{args.camera_dir.parent.name}/{args.camera_dir.name}"
    case_id = f"gt{args.gt_id}_{selected[0]}_{selected[-1]}"
    images_dir = args.output_dir / "cases" / run_id / case_id
    images_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.camera_dir / "vdo.avi"
    capture = cv2.VideoCapture(str(video_path))
    observations = []
    decoded = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"영상을 열지 못했습니다. {video_path}")
        for frame_id in selected:
            # 순서대로 건너뛰며 GT의 1번 프레임과 영상의 첫 프레임을 맞춘다.
            while decoded < frame_id - 1:
                if not capture.grab():
                    raise RuntimeError(f"영상에서 {frame_id}번 프레임에 도달할 수 없습니다.")
                decoded += 1
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"{frame_id}번 프레임을 읽지 못했습니다.")
            decoded += 1
            gt_boxes = gt[frame_id]
            high = describe_track(args.gt_id, gt_boxes, high_tracks.get(frame_id, []), args.min_iou)
            two = describe_track(args.gt_id, gt_boxes, two_tracks.get(frame_id, []), args.min_iou)
            gt_box = dict(gt_boxes)[args.gt_id]
            image = comparison_image(frame, frame_id, args.gt_id, gt_box, high, two, args.min_iou)
            image_path = images_dir / f"frame_{frame_id:06d}.jpg"
            # imencode 뒤 Path로 쓰면 Windows 한글 경로에서도 저장할 수 있다.
            encoded, data = cv2.imencode(".jpg", image)
            if not encoded:
                raise RuntimeError(f"이미지를 인코딩하지 못했습니다. {image_path}")
            image_path.write_bytes(data.tobytes())
            observations.append({"frame_id": frame_id, "gt_box": gt_box, "high_only": high,
                                 "two_stage": two, "image_path": image_path.relative_to(args.output_dir).as_posix()})
            print(f"{frame_id}: high={high['status']} #{high['track_id']} IoU={high['iou']}, "
                  f"two={two['status']} #{two['track_id']} IoU={two['iou']}")
    finally:
        capture.release()

    print(f"이미지 폴더: {images_dir.resolve()}")
    if args.observation:
        gained = sum(r["high_only"]["status"] != "matched" and r["two_stage"]["status"] == "matched" for r in observations)
        lost = sum(r["high_only"]["status"] == "matched" and r["two_stage"]["status"] != "matched" for r in observations)
        case_type = {(True, True): "mixed", (True, False): "improved",
                     (False, True): "regressed", (False, False): "unchanged"}[(gained > 0, lost > 0)]
        case = {"run_id": run_id, "case_id": case_id, "camera": camera, "gt_id": args.gt_id,
                "frame_range": [selected[0], selected[-1]], "gained_matches": gained, "lost_matches": lost,
                "case_type": case_type, "min_iou": args.min_iou, "observation": args.observation,
                "image_paths": [r["image_path"] for r in observations], "frames": observations}
        path = args.output_dir / "cases.jsonl"
        save_case(path, case)
        print(f"사례 저장: {path.resolve()}")


if __name__ == "__main__":
    main()
