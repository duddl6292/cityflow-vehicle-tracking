import argparse
import csv
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(
        description = "지정한 프레임에 GT 차량 박스를 그립니다."
    )

    parser.add_argument(
        "--camera-dir",
        type = Path,
        required = True,
    )

    parser.add_argument(
        "--frame-id",
        type = int,
        required = True,
    )

    parser.add_argument(
        "--output-dir",
        type = Path,
        default = Path(__file__).resolve().parents[1] / "outputs",
    )

    args = parser.parse_args()

    if args.frame_id < 1:
        parser.error("--frame-id는 1 이상이어야 합니다.")

    video_path = args.camera_dir / "vdo.avi"
    gt_path = args.camera_dir / "gt" / "gt.txt"

    for path in (video_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(f"파일이 없습니다: {path}")

    boxes = []

    with gt_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            frame_id = int(row[0])

            if frame_id == args.frame_id:
                vehicle_id = int(row[1])

                x, y, w, h = [
                    int(float(value)) for value in row[2:6]
                ]

                boxes.append((vehicle_id, x, y, w, h))

    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        for frame_index in range(args.frame_id):
            success, frame = cap.read()

            if not success:
                raise RuntimeError(
                    f"{frame_index + 1}번째 프레임을 읽지 못했습니다."
                )

    finally:
        cap.release()

    for vehicle_id, x, y, w, h in boxes:
        cv2.rectangle(
            frame,                  
            (x, y),                 
            (x + w - 1, y + h - 1), 
            (0, 255, 0),            
            2,                      
        )

        cv2.putText(
            frame,
            f"GT ID: {vehicle_id}",
            (x, max(y - 10, 20)),     
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,                     
            (0, 255, 0),             
            2,                       
        )

    scenario = args.camera_dir.parent.name
    camera = args.camera_dir.name

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / (
        f"{scenario}_{camera}_frame_{args.frame_id:06d}_gt.jpg"
    )

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError("GT 이미지를 저장하지 못했습니다.")

    print(f"선택한 프레임: {args.frame_id}")
    print(f"정답 박스 수: {len(boxes)}")
    print(f"이미지 저장: {output_path.resolve()}")


if __name__ == "__main__":
    main()