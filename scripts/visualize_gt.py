# 터미널 실행 옵션을 읽는 라이브러리
import argparse

# 쉼표로 구분된 GT 파일을 읽는 파이썬 기본 라이브러리
import csv

# 파일과 폴더 경로를 다루는 클래스
from pathlib import Path

# 영상 읽기와 박스 그리기에 사용할 OpenCV
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="지정한 프레임에 GT 차량 박스를 그립니다."
    )

    # 처리할 카메라 폴더를 입력받는다.
    parser.add_argument(
        "--camera-dir",
        type=Path,
        required=True,
    )

    # 확인할 프레임 번호를 정수로 입력받는다.
    # 이 데이터의 GT 프레임 번호는 1부터 시작한다.
    parser.add_argument(
        "--frame-id",
        type=int,
        required=True,
    )

    # 결과 저장 위치: 기본값은 프로젝트의 outputs 폴더
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )

    args = parser.parse_args()

    # 0이나 음수는 GT 프레임 번호로 사용할 수 없다.
    if args.frame_id < 1:
        parser.error("--frame-id는 1 이상이어야 합니다.")

    video_path = args.camera_dir / "vdo.avi"
    gt_path = args.camera_dir / "gt" / "gt.txt"

    for path in (video_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(f"파일이 없습니다: {path}")

    # 선택한 프레임에 등장하는 차량들의 정답 정보를 담을 리스트
    boxes = []

    with gt_path.open(encoding="utf-8-sig", newline="") as file:
        # 한 줄을 쉼표 기준으로 분리해서 문자열 리스트로 반환한다.
        # 예: ["55", "34", "1581", "346", ...]
        reader = csv.reader(file)

        for row in reader:
            # 빈 줄은 건너뛴다.
            if not row:
                continue

            # 첫 번째 열은 프레임 번호다.
            frame_id = int(row[0])

            # 선택한 프레임의 정답만 모은다.
            if frame_id == args.frame_id:
                vehicle_id = int(row[1])

                # row[2:6]은 인덱스 2, 3, 4, 5의 값을 가져온다.
                # 각각 left, top, width, height에 해당한다.
                #
                # float → int 순서로 변환하면
                # "1581"과 "1581.0" 같은 표기를 모두 처리할 수 있다.
                x, y, w, h = [
                    int(float(value)) for value in row[2:6]
                ]

                # 차량 한 대의 정보를 튜플로 묶어 저장한다.
                boxes.append((vehicle_id, x, y, w, h))

    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        # 첫 프레임부터 목표 프레임까지 순서대로 읽는다.
        #
        # --frame-id 55라면 read()를 55번 호출한다.
        # 마지막으로 읽은 frame이 55번째 프레임이 된다.
        for frame_index in range(args.frame_id):
            success, frame = cap.read()

            if not success:
                raise RuntimeError(
                    f"{frame_index + 1}번째 프레임을 읽지 못했습니다."
                )

    finally:
        # 영상 파일을 닫아도 이미 읽어온 frame 배열은 사용할 수 있다.
        cap.release()

    # 선택한 프레임의 모든 정답 박스를 그린다.
    for vehicle_id, x, y, w, h in boxes:
        cv2.rectangle(
            frame,                  # 박스를 그릴 이미지
            (x, y),                 # 왼쪽 위 좌표
            (x + w - 1, y + h - 1), # 마지막 포함 픽셀의 좌표
            (0, 255, 0),            # BGR 색상: 초록색
            2,                      # 선 두께: 2픽셀
        )

        # 박스 위에 차량 ID를 표시한다.
        cv2.putText(
            frame,
            f"GT ID: {vehicle_id}",
            (x, max(y - 10, 20)),     # 글자가 위쪽 화면 밖으로 나가지 않게 조정
            cv2.FONT_HERSHEY_SIMPLEX, # 글꼴
            0.7,                     # 글자 크기 배율
            (0, 255, 0),             # 글자 색상
            2,                       # 글자 선 두께
        )

    scenario = args.camera_dir.parent.name
    camera = args.camera_dir.name

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 프레임 번호까지 이름에 넣어 다른 프레임의 결과와 구분한다.
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