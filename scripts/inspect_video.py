import argparse
from pathlib import Path
import cv2

def main():
    parser = argparse.ArgumentParser(
        description="카메라 영상 정보와 GT를 확인합니다."
    )

    parser.add_argument(
        "--camera-dir",
        type = Path,
        required = True,
        help = "vdo.avi와 gt 폴더가 들어있는 카메라 폴더 경로",
    )

    parser.add_argument(
        "--output-dir",
        type = Path,
        default = Path(__file__).resolve().parents[1] / "outputs",
        help = "결과를 저장할 폴더. 기본값: 프로젝트의 outputs 폴더"
    )

    args = parser.parse_args()

    camera_dir = args.camera_dir

    video_path = camera_dir / "vdo.avi"
    gt_path = camera_dir / "gt" / "gt.txt"

    for path in (video_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(f"파일이 없습니다: {path}")
    
    scenario = camera_dir.parent.name
    camera = camera_dir.name
    
    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"대상: {scenario}/{camera}")
        print(f"해상도: {width} * {height}")
        print(f"FPS: {fps}")
        print(f"프레임 수: {frame_count}")
        
        if fps > 0:
            print(f"영상 길이: 약 {frame_count / fps:.1f}초")
        
        # read(): VideoCapture 클래스의 메서드
        # 프레임 읽기 성공 여부와 실제 이미지 데이터를 반환함
        success, frame = cap.read() 
        
        if not success:
            raise RuntimeError("첫 프레임을 읽을 수 없습니다.")
        
        args.output_dir.mkdir(parents = True, exist_ok = True)
        
        output_path = (
            args.output_dir / f"{scenario}_{camera}_first_frame.jpg"
        )
        
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError("첫 프레임을 저장하지 못했습니다.")
        
        print(f"이미지 저장: {output_path.resolve()}")
        
    finally:
        cap.release()
    
    print("\nGT 앞 5줄:")
    with gt_path.open(encoding = "utf-8-sig") as file:
        for _, line in zip(range(5), file):
            print(line.strip())
            
if __name__ == "__main__":
    main()