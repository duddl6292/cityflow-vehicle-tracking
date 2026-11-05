import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2


def load_tracks(csv_path):
    """추적 결과를 프레임 번호별로 묶습니다."""

    tracks = defaultdict(list)

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            frame_id = int(row["frame_id"])
            track_id = int(row["track_id"])

            box = [
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ]

            tracks[frame_id].append((track_id, box))

    return dict(tracks)


def get_color(track_id):
    """같은 ID에는 항상 같은 색을 반환합니다. OpenCV는 BGR 순서입니다."""

    return (
        80 + (track_id * 37) % 176,
        80 + (track_id * 67) % 176,
        80 + (track_id * 97) % 176,
    )


def main():
    parser = argparse.ArgumentParser(
        description="추적 CSV의 박스와 ID를 원본 영상에 표시합니다."
    )

    parser.add_argument(
        "--tracks-csv",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    tracks = load_tracks(args.tracks_csv)

    # 추적 CSV와 같은 폴더의 실행 기록을 읽습니다.
    run_path = args.tracks_csv.parent / "run.json"

    with run_path.open(encoding="utf-8") as file:
        run_info = json.load(file)

    video_path = Path(run_info["source_video"])
    total_frames = int(run_info["processed_frames"])

    if total_frames < 1:
        raise ValueError("처리할 프레임이 없습니다.")

    # 파일 이름은 유지하고 확장자만 .mp4로 바꿉니다.
    output_path = args.tracks_csv.with_suffix(".mp4")

    cap = cv2.VideoCapture(str(video_path))
    writer = None

    try:
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"잘못된 영상 FPS입니다: {fps}")

        for frame_id in range(1, total_frames + 1):
            success, frame = cap.read()

            if not success:
                raise RuntimeError(
                    f"{frame_id}번 프레임을 읽지 못했습니다."
                )

            height, width = frame.shape[:2]

            # 첫 프레임에서 영상 크기를 확인한 뒤 저장기를 만듭니다.
            if writer is None:
                frame_size = (width, height)

                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    frame_size,
                )

                if not writer.isOpened():
                    raise RuntimeError("출력 영상을 생성하지 못했습니다.")

            elif (width, height) != frame_size:
                raise RuntimeError("영상 도중 프레임 크기가 바뀌었습니다.")

            frame_tracks = tracks.get(frame_id, [])

            for track_id, box in frame_tracks:
                # CSV의 실수 좌표는 그대로 두고, 그릴 때만 정수로 바꿉니다.
                x1, y1, x2, y2 = [
                    int(round(value))
                    for value in box
                ]

                color = get_color(track_id)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                # ID 글자가 화면 밖으로 나가지 않도록 위치를 제한합니다.
                text_x = max(0, min(x1, width - 140))
                text_y = max(25, min(y1 - 8, height - 10))

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"Frame {frame_id} | Tracks {len(frame_tracks)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 트랙이 없는 프레임도 저장해서 시간 흐름을 유지합니다.
            writer.write(frame)

            if frame_id % 100 == 0 or frame_id == total_frames:
                print(f"영상 저장 중: {frame_id}/{total_frames}")

    finally:
        # 오류가 나더라도 영상 자원을 정리합니다.
        cap.release()

        if writer is not None:
            writer.release()

    print(f"완료: {total_frames}프레임")
    print(f"추적 영상: {output_path.resolve()}")


if __name__ == "__main__":
    main()