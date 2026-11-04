"""YOLO11s로 카메라 영상 전체를 순차 추론하고 CSV·영상·실행 정보를 저장한다."""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import torch
from ultralytics import YOLO, __version__ as ultralytics_version


# 프로젝트/scripts/detect_video.py에 파일을 두는 것을 기준으로 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = [
    "frame_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
]


def parse_args():
    """터미널에서 카메라 경로와 탐지 설정을 읽는다."""
    parser = argparse.ArgumentParser(description="YOLO11s로 영상 전체의 차량을 탐지합니다.")
    parser.add_argument("--camera-dir", type=Path, required=True,
                        help="vdo.avi가 들어 있는 카메라 폴더")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="정사각형 추론 입력의 한 변 길이")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="저장할 예측의 최소 신뢰도")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "detections")
    args = parser.parse_args()
    if args.imgsz < 1:
        parser.error("--imgsz는 1 이상이어야 합니다.")
    if not 0 < args.conf < 1:
        parser.error("--conf는 0보다 크고 1보다 작아야 합니다.")
    return args


def save_run_info(path, info):
    """설정과 처리 상태를 JSON으로 저장한다. 한글도 그대로 기록한다."""
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def run_detection(args):
    """한 번의 실행: 모델 로딩 → 전체 프레임 추론 → 결과 저장."""
    video_path = args.camera_dir / "vdo.avi"
    if not video_path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {video_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다. PyTorch 설치를 확인하세요.")

    # 모델은 프레임마다 다시 만들지 않고, 전체 영상에서 한 번 불러와 재사용한다.
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "yolo11s.pt"
    model = YOLO(str(weights_path))

    # detect_frame.py와 동일한 탐지 조건이다. classes는 COCO의 car·bus·truck.
    prediction_settings = dict(
        device=0, imgsz=args.imgsz, rect=False, conf=args.conf,
        iou=0.7, classes=[2, 5, 7], verbose=False,
    )

    # 실행마다 새 폴더를 만들어 이전 실험 결과가 덮어써지지 않게 한다.
    started_at = datetime.now().astimezone()
    scenario, camera = args.camera_dir.parent.name, args.camera_dir.name
    run_dir = args.output_dir / (
        f"{scenario}_{camera}_{started_at.strftime('%Y%m%d_%H%M%S_%f')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "detections.csv"
    output_video = run_dir / "detections.mp4"
    info_path = run_dir / "run.json"

    info = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "source_video": str(video_path.resolve()),
        "scenario": scenario, "camera": camera,
        "model": "yolo11s.pt", "weights_path": str(weights_path.resolve()),
        "prediction_settings": prediction_settings,
        "python": sys.version.split()[0], "opencv": cv2.__version__,
        "pytorch": torch.__version__, "ultralytics": ultralytics_version,
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "frame_id_base": 1, "box_format": "xyxy_original_pixels",
        "video_codec": "mp4v",
        "source_fps": None, "reported_frames": None, "frame_size": None,
        "processed_frames": 0, "total_detections": 0,
    }
    # 강제 종료되더라도 실행 설정과 진행 중 상태가 남도록 먼저 기록한다.
    save_run_info(info_path, info)
    print(f"대상: {scenario}/{camera}")
    print(f"출력 폴더: {run_dir.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    video_writer = None
    processed_frames = 0
    total_detections = 0
    started = perf_counter()

    try:
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("원본 FPS를 읽지 못했습니다. 영상 정보를 확인하세요.")
        reported = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frames = int(reported) if math.isfinite(reported) and reported > 0 else 0
        info.update(source_fps=fps, reported_frames=expected_frames or None)
        save_run_info(info_path, info)

        with csv_path.open("w", encoding="utf-8", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(CSV_COLUMNS)

            # 한 프레임을 읽고 저장한 뒤 다음으로 이동한다. 전체 결과를 쌓지 않는다.
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_id = processed_frames + 1  # GT와 같이 1부터 번호를 매긴다.
                height, width = frame.shape[:2]

                if video_writer is None:
                    # 출력 크기는 모델 입력 640이 아닌 원본 프레임 크기다.
                    # 재생 FPS도 원본 값을 사용한다. 추론 처리 속도와는 별개다.
                    info["frame_size"] = [width, height]
                    video_writer = cv2.VideoWriter(
                        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (width, height),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError("MP4 저장기를 열지 못했습니다. 코덱을 확인하세요.")
                elif info["frame_size"] != [width, height]:
                    raise RuntimeError(f"{frame_id}번째 프레임의 크기가 바뀌었습니다.")

                # **는 딕셔너리의 설정을 개별 인자로 전달한다. 가중치 학습은 하지 않는다.
                result = model.predict(source=frame, **prediction_settings)[0]
                coordinates = result.boxes.xyxy.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                class_ids = result.boxes.cls.cpu().tolist()

                # 표시용 복사본에만 박스를 그린다. 다음 추론에는 원본을 넣는다.
                annotated = frame.copy()
                for xyxy, confidence, class_id in zip(coordinates, confidences, class_ids):
                    class_id = int(class_id)
                    csv_writer.writerow([
                        frame_id, class_id, result.names[class_id], confidence, *xyxy,
                    ])  # CSV에는 원본 좌표와 신뢰도를 소수점 그대로 저장한다.

                    # 그림을 그릴 때만 정수로 변환한다. 글자는 작게 표시한다.
                    x1, y1, x2, y2 = [round(value) for value in xyxy]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(
                        annotated, f"{result.names[class_id]} {confidence:.2f}",
                        (max(x1, 0), max(y1 - 4, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1,
                    )

                # 검출이 0개인 프레임도 영상에는 반드시 저장한다.
                cv2.putText(
                    annotated, f"Frame {frame_id} | Detections {len(coordinates)}",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
                )
                video_writer.write(annotated)
                processed_frames += 1
                total_detections += len(coordinates)
                if processed_frames % 100 == 0:
                    total_label = str(expected_frames) if expected_frames else "?"
                    print(f"처리 중: {processed_frames}/{total_label} 프레임", flush=True)

        if processed_frames == 0:
            raise RuntimeError("영상에서 프레임을 읽지 못했습니다.")
        # EOF와 중간 디코딩 실패를 구분하는 보조 확인이다. 메타데이터 오차도 가능하다.
        if expected_frames and processed_frames != expected_frames:
            raise RuntimeError(
                f"보고된 프레임 {expected_frames}개와 읽은 프레임 {processed_frames}개가 "
                "다릅니다. 메타데이터 또는 영상 읽기 상태를 확인하세요."
            )
        info["status"] = "completed"
    except BaseException as error:
        # 중단·오류를 성공 결과로 착각하지 않도록 기록하고 원래 예외를 다시 발생시킨다.
        info["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        info["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()  # 파일 끝 정보를 기록해야 재생할 수 있다.
        elapsed = perf_counter() - started
        info.update(
            processed_frames=processed_frames, total_detections=total_detections,
            processing_seconds=elapsed,
            pipeline_fps=processed_frames / elapsed if elapsed > 0 else None,
        )
        save_run_info(info_path, info)

    # 이 속도는 읽기·추론·그리기·CSV/영상 저장을 포함하며 모델 로딩은 제외한다.
    print(f"완료: {processed_frames} 프레임, 총 검출 {total_detections}개")
    print(f"전체 처리 시간: {elapsed:.1f}초 / 처리 속도: {info['pipeline_fps']:.2f} FPS")
    print(f"CSV: {csv_path.resolve()}")
    print(f"탐지 영상: {output_video.resolve()}")
    print(f"실행 정보: {info_path.resolve()}")
    return run_dir


if __name__ == "__main__":
    run_detection(parse_args())
