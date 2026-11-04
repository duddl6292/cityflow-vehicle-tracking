"""원본 영상의 지정 프레임을 YOLO11s로 탐지하고 이미지와 CSV를 저장한다."""

import argparse
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# 이 파일을 프로젝트/scripts/detect_frame.py에 두는 것을 기준으로 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_frame(video_path, frame_id):
    """1부터 시작하는 프레임 번호를 받아 원본 이미지 한 장을 반환한다."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        # 처음부터 차례로 읽는다. 55번 호출하면 마지막 이미지가 55번째다.
        for index in range(frame_id):
            success, frame = cap.read()
            if not success:
                raise RuntimeError(f"{index + 1}번째 프레임을 읽지 못했습니다.")
        return frame
    finally:
        # return이나 예외가 발생해도 영상 파일 자원은 정리한다.
        cap.release()


def main():
    # 실행할 때 바꿀 수 있는 입력값을 정의한다.
    parser = argparse.ArgumentParser(description="원본 프레임의 차량을 YOLO11s로 탐지합니다.")
    parser.add_argument("--camera-dir", type=Path, required=True,
                        help="vdo.avi가 들어 있는 카메라 폴더")
    parser.add_argument("--frame-id", type=int, required=True,
                        help="1부터 시작하는 영상 프레임 번호")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="추론 입력의 한 변 길이. 기본값: 640")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="저장할 예측의 최소 신뢰도. 기본값: 0.1")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "detections")
    args = parser.parse_args()

    if args.frame_id < 1 or args.imgsz < 1:
        parser.error("--frame-id와 --imgsz는 1 이상이어야 합니다.")
    if not 0 < args.conf < 1:
        parser.error("--conf는 0보다 크고 1보다 작아야 합니다.")

    video_path = args.camera_dir / "vdo.avi"
    if not video_path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {video_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다. PyTorch 설치를 확인하세요.")

    # GT 박스를 그린 이미지 대신 영상에서 읽은 원본을 입력으로 사용한다.
    frame = read_frame(video_path, args.frame_id)

    # 첫 실행 시 공식 사전학습 가중치를 다운로드하고, 이후에는 재사용한다.
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(exist_ok=True)
    model = YOLO(str(weights_dir / "yolo11s.pt"))

    # COCO 클래스 번호: 2=car, 5=bus, 7=truck.
    # 이번 확인에서는 이 세 클래스를 차량으로 취급한다.
    result = model.predict(
        source=frame,       # OpenCV가 읽은 원본 BGR 이미지
        device=0,           # 첫 번째 NVIDIA GPU 사용
        imgsz=args.imgsz,    # 모델에 전달할 입력 크기
        rect=False,         # 비율을 유지해 축소/확대하고 정사각형으로 패딩
        conf=args.conf,     # 이 값보다 낮은 신뢰도의 박스는 제외
        iou=0.7,            # 겹치는 예측을 정리하는 NMS의 IoU 임계값
        classes=[2, 5, 7],  # 차량 클래스만 결과로 남김
        verbose=False,     # 라이브러리의 상세 출력 생략
    )[0]                    # 이미지 한 장을 입력했으므로 첫 번째 결과 사용

    scenario = args.camera_dir.parent.name
    camera = args.camera_dir.name
    stem = f"{scenario}_{camera}_frame_{args.frame_id:06d}_yolo11s"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"{stem}_pred.jpg"
    csv_path = args.output_dir / f"{stem}_pred.csv"

    # plot()은 예측 박스·클래스·신뢰도를 표시한 BGR 이미지를 반환한다.
    if not cv2.imwrite(str(image_path), result.plot()):
        raise RuntimeError(f"예측 이미지를 저장하지 못했습니다: {image_path}")

    # GPU에 있는 결과를 CPU로 옮겨 Python 리스트로 변환한다.
    # xyxy는 원본 이미지 기준의 왼쪽 위·오른쪽 아래 픽셀 좌표다.
    # 계산용 좌표와 신뢰도는 소수점을 유지해 저장한다.
    coordinates = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    class_ids = result.boxes.cls.cpu().tolist()

    # newline=""는 Windows의 CSV 줄바꿈 처리를 위해 사용한다.
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame_id", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2",
        ])
        # 같은 차량 예측에 해당하는 좌표·신뢰도·클래스를 함께 가져온다.
        for xyxy, confidence, class_id in zip(coordinates, confidences, class_ids):
            class_id = int(class_id)
            # *xyxy는 좌표 리스트의 네 값을 개별 열로 펼친다.
            writer.writerow([
                args.frame_id, class_id, result.names[class_id], confidence, *xyxy,
            ])

    # 예측이 없으면 박스 없는 이미지와 헤더만 있는 CSV가 저장된다.
    print(f"대상: {scenario}/{camera}, 프레임: {args.frame_id}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"입력 크기: {args.imgsz}, 최소 신뢰도: {args.conf}, NMS IoU: 0.7")
    print(f"차량 예측 수: {len(coordinates)}")
    print(f"예측 이미지: {image_path.resolve()}")
    print(f"예측 CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
