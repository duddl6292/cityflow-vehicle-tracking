import argparse
import csv
from collections import defaultdict
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import numpy as np
import json


def load_detections(csv_path):
    """탐지 CSV를 읽어서 프레임 번호별로 묶어 반환합니다."""

    # 일반 딕셔너리와 달리, 처음 접근하는 키에 빈 리스트를 만들어 줍니다.
    # 예: detections[55]에 처음 접근하면 []가 자동으로 생성됩니다.
    detections = defaultdict(list)

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        # 첫 줄의 열 이름을 키로 사용해서 각 행을 읽습니다.
        # 예: row["frame_id"], row["confidence"]
        reader = csv.DictReader(file)

        for row in reader:
            # CSV에서 읽은 값은 문자열이므로 숫자로 변환합니다.
            frame_id = int(row["frame_id"])

            detection = {
                "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]),

                # 좌표는 소수점을 유지합니다.
                # 이후 IoU 계산과 추적에 그대로 사용합니다.
                "bbox": [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ],
            }

            # 같은 프레임 번호의 리스트에 탐지 결과를 추가합니다.
            detections[frame_id].append(detection)

    # 읽기가 끝났으므로 일반 딕셔너리로 변환해 반환합니다.
    return dict(detections)

def calculate_iou(box_a, box_b):
    """xyxy 형식의 두 박스를 받아 IoU를 반환합니다."""

    # 각 박스의 왼쪽 위, 오른쪽 아래 좌표를 꺼냅니다.
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 교집합의 왼쪽 위는 두 박스 중 더 안쪽에 있는 좌표입니다.
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    # 교집합의 오른쪽 아래도 더 안쪽에 있는 좌표입니다.
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    # 겹치지 않으면 차이가 음수가 될 수 있으므로 0으로 제한합니다.
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    # 각 박스의 넓이를 계산합니다.
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # 두 넓이를 더하면 교집합을 두 번 세므로 한 번 뺍니다.
    union_area = area_a + area_b - intersection_area

    # 두 박스 모두 넓이가 0인 경우, 0으로 나누는 것을 막습니다.
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

def split_detections(
    frame_detections,
    high_threshold=0.6,
    low_threshold=0.1,
):
    """한 프레임의 탐지 결과를 신뢰도에 따라 나눕니다."""

    high_detections = []
    low_detections = []

    for detection in frame_detections:
        confidence = detection["confidence"]

        if confidence >= high_threshold:
            high_detections.append(detection)

        elif confidence >= low_threshold:
            low_detections.append(detection)

    # 리스트 두 개를 함께 반환합니다.
    return high_detections, low_detections

def build_iou_matrix(track_boxes, detection_boxes):
    """각 트랙 박스와 각 탐지 박스 사이의 IoU를 계산합니다."""

    matrix = []

    # 트랙 하나가 행 하나에 해당합니다.
    for track_box in track_boxes:
        row = []

        # 해당 트랙과 모든 탐지 박스를 각각 비교합니다.
        for detection_box in detection_boxes:
            iou = calculate_iou(track_box, detection_box)
            row.append(iou)

        matrix.append(row)

    return matrix

def match_boxes(track_boxes, detection_boxes, min_iou=0.3):
    """박스를 일대일로 연결하고, 연결되지 않은 인덱스도 반환합니다."""

    num_tracks = len(track_boxes)
    num_detections = len(detection_boxes)

    # 한쪽에 박스가 없으면 연결할 수 없습니다.
    if num_tracks == 0 or num_detections == 0:
        return [], list(range(num_tracks)), list(range(num_detections))

    iou_matrix = build_iou_matrix(track_boxes, detection_boxes)

    # 할당 함수는 비용이 작은 조합을 선택합니다.
    # 따라서 IoU가 클수록 비용이 작아지도록 1 - IoU로 바꿉니다.
    #
    # 기준 미달 조합에는 큰 비용을 줍니다.
    # 이 값은 정상 연결들의 총비용보다 커서,
    # 기준을 만족하는 연결 수를 우선 늘리도록 합니다.
    penalty = min(num_tracks, num_detections) + 1

    cost_matrix = []

    for row in iou_matrix:
        cost_row = []

        for iou in row:
            if iou >= min_iou:
                cost_row.append(1.0 - iou)
            else:
                cost_row.append(penalty)

        cost_matrix.append(cost_row)

    # 전체 비용을 최소화하는 행·열 인덱스 쌍을 구합니다.
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_tracks = set()
    matched_detections = set()

    for track_index, detection_index in zip(
        track_indices, detection_indices
    ):
        # 반환된 인덱스를 일반 Python 정수로 바꿉니다.
        track_index = int(track_index)
        detection_index = int(detection_index)

        # 큰 비용을 준 조합이 반환되더라도 실제 연결로 채택하지 않습니다.
        if iou_matrix[track_index][detection_index] < min_iou:
            continue

        matches.append((track_index, detection_index))
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)

    unmatched_tracks = [
        index
        for index in range(num_tracks)
        if index not in matched_tracks
    ]

    unmatched_detections = [
        index
        for index in range(num_detections)
        if index not in matched_detections
    ]

    return matches, unmatched_tracks, unmatched_detections

class Track:
    """차량 하나의 ID, 추정 위치, 움직임 정보를 관리합니다."""

    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.missed_frames = 0

        # 트랙마다 자신의 상태와 불확실성을 관리합니다.
        self.kalman_filter = BoxKalmanFilter()

        self.mean, self.covariance = self.kalman_filter.initiate(bbox)

        # IoU 계산에 사용할 xyxy 좌표입니다.
        self.bbox = xyah_to_xyxy(self.mean[:4])

    def predict(self):
        """다음 프레임의 위치를 예측합니다."""

        self.mean, self.covariance = self.kalman_filter.predict(
            self.mean,
            self.covariance,
        )

        self.bbox = xyah_to_xyxy(self.mean[:4])

    def update(self, detection_box):
        """매칭된 탐지로 위치와 속도를 보정합니다."""

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            detection_box,
        )

        self.bbox = xyah_to_xyxy(self.mean[:4])
        self.missed_frames = 0

    def mark_missed(self):
        """이번 프레임에서 탐지와 연결되지 않았음을 기록합니다."""

        self.missed_frames += 1

def xyxy_to_xyah(box):
    """모서리 좌표를 중심 좌표, 가로세로 비율, 높이로 변환합니다."""

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    # 높이가 0이면 비율을 계산할 수 없습니다.
    if width <= 0 or height <= 0:
        raise ValueError("박스의 너비와 높이는 양수여야 합니다.")

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    aspect_ratio = width / height

    return [center_x, center_y, aspect_ratio, height]


def xyah_to_xyxy(box):
    """중심 좌표, 가로세로 비율, 높이를 모서리 좌표로 변환합니다."""

    center_x, center_y, aspect_ratio, height = box

    # 비율 = 너비 / 높이이므로, 너비 = 비율 × 높이입니다.
    width = aspect_ratio * height

    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    return [x1, y1, x2, y2]

class BoxKalmanFilter:
    """박스의 위치와 크기를 일정한 속도 모델로 예측합니다."""

    def __init__(self):
        # 상태 8개를 다음 프레임의 상태로 변환하는 행렬입니다.
        self.motion_matrix = np.eye(8)

        # 위치·크기에 각각의 변화량을 더하도록 설정합니다.
        # cx_next = cx + vx
        # cy_next = cy + vy
        # a_next = a + va
        # h_next = h + vh
        self.motion_matrix[:4, 4:] = np.eye(4)

        # ByteTrack 공식 구현에서 사용하는 불확실성 설정값입니다.
        # 위치와 속도의 오차 크기를 박스 높이에 비례하게 설정합니다.
        self.position_weight = 1.0 / 20
        self.velocity_weight = 1.0 / 160

    def initiate(self, box):
        """첫 탐지 박스로 상태와 불확실성을 초기화합니다."""

        measurement = np.array(
            xyxy_to_xyah(box),
            dtype=float,
        )

        # 첫 탐지만으로는 이동 속도를 알 수 없으므로 0으로 시작합니다.
        velocity = np.zeros(4)

        # 위치·크기 4개와 속도 4개를 이어 붙입니다.
        mean = np.concatenate([measurement, velocity])

        height = measurement[3]

        # 각 상태의 초기 표준편차입니다.
        std = np.array([
            2 * self.position_weight * height,   # cx
            2 * self.position_weight * height,   # cy
            1e-2,                               # a
            2 * self.position_weight * height,   # h
            10 * self.velocity_weight * height,  # vx
            10 * self.velocity_weight * height,  # vy
            1e-5,                               # va
            10 * self.velocity_weight * height,  # vh
        ])

        # 표준편차를 제곱하면 분산입니다.
        # 초기에는 상태 사이의 공분산을 0으로 설정합니다.
        covariance = np.diag(std ** 2)

        return mean, covariance

    def predict(self, mean, covariance):
        """한 프레임 뒤의 상태와 불확실성을 예측합니다."""

        height = mean[3]

        # 일정한 속도로 움직인다는 가정에도 오차가 있으므로
        # 예측 과정에서 추가되는 불확실성을 설정합니다.
        std = np.array([
            self.position_weight * height,
            self.position_weight * height,
            1e-2,
            self.position_weight * height,
            self.velocity_weight * height,
            self.velocity_weight * height,
            1e-5,
            self.velocity_weight * height,
        ])

        process_noise = np.diag(std ** 2)

        # @는 행렬 곱셈입니다.
        predicted_mean = self.motion_matrix @ mean

        # 위치뿐 아니라 불확실성도 다음 프레임으로 전달합니다.
        # .T는 행렬의 행과 열을 바꾸는 전치입니다.
        predicted_covariance = (
            self.motion_matrix
            @ covariance
            @ self.motion_matrix.T
            + process_noise
        )

        return predicted_mean, predicted_covariance

    def update(self, mean, covariance, detection_box):
        """연결된 탐지 결과로 예측 상태와 불확실성을 보정합니다."""

        # 탐지 좌표를 Kalman Filter가 사용하는 xyah로 변환합니다.
        measurement = np.array(
            xyxy_to_xyah(detection_box),
            dtype=float,
        )

        # 상태 8개 중 직접 관측할 수 있는 앞의 4개를 꺼내는 행렬입니다.
        # 탐지로는 cx, cy, a, h를 관측하고, 속도는 추정합니다.
        observation_matrix = np.eye(4, 8)

        height = mean[3]

        # 탐지 좌표 자체에도 오차가 있다고 가정합니다.
        # ByteTrack 공식 구현과 같은 설정값입니다.
        std = np.array([
            self.position_weight * height,
            self.position_weight * height,
            1e-1,
            self.position_weight * height,
        ])

        measurement_noise = np.diag(std ** 2)

        # 예측 상태를 관측과 같은 4차원 형식으로 바꿉니다.
        projected_mean = observation_matrix @ mean

        # 예측의 불확실성과 관측 오차를 합칩니다.
        projected_covariance = (
            observation_matrix
            @ covariance
            @ observation_matrix.T
            + measurement_noise
        )

        # Kalman Gain은 관측과의 차이를 얼마나 반영할지 결정합니다.
        # 역행렬을 직접 구하는 대신 연립방정식을 풉니다.
        cross_covariance = covariance @ observation_matrix.T

        kalman_gain = np.linalg.solve(
            projected_covariance,
            cross_covariance.T,
        ).T

        # 실제 탐지와 예측의 차이입니다.
        innovation = measurement - projected_mean

        # 차이를 반영해 위치·크기와 속도를 함께 보정합니다.
        updated_mean = mean + kalman_gain @ innovation

        updated_covariance = (
            covariance
            - kalman_gain
            @ projected_covariance
            @ kalman_gain.T
        )

        # 부동소수점 계산 오차로 생기는 비대칭을 정리합니다.
        updated_covariance = (
            updated_covariance + updated_covariance.T
        ) / 2

        return updated_mean, updated_covariance

class HighConfidenceTracker:
    """높은 신뢰도 탐지만 사용하는 비교 기준 추적기입니다."""

    def __init__(
        self,
        high_threshold=0.6,
        min_iou=0.3,
        max_missed=20,
    ):
        self.high_threshold = high_threshold
        self.min_iou = min_iou

        # 연결되지 않은 트랙을 보관할 최대 프레임 수입니다.
        # 현재 영상은 10 FPS이므로 20프레임은 약 2초입니다.
        self.max_missed = max_missed

        self.tracks = []
        self.next_track_id = 1

    def update(self, frame_detections):
        """한 프레임을 처리하고, 이번에 관측된 트랙들을 반환합니다."""

        # 이번 기준 추적기에서는 높은 신뢰도 탐지만 사용합니다.
        high_detections, _ = split_detections(
            frame_detections,
            high_threshold=self.high_threshold,
        )

        detection_boxes = [
            detection["bbox"]
            for detection in high_detections
        ]

        # 1. 기존 트랙들의 위치를 예측합니다.
        for track in self.tracks:
            track.predict()

        predicted_boxes = [
            track.bbox
            for track in self.tracks
        ]

        # 2. 예측 위치와 현재 탐지를 연결합니다.
        matches, unmatched_tracks, unmatched_detections = match_boxes(
            predicted_boxes,
            detection_boxes,
            min_iou=self.min_iou,
        )

        for track_index, detection_index in matches:
            self.tracks[track_index].update(
                detection_boxes[detection_index]
            )

        # 3. 연결되지 않은 트랙은 바로 지우지 않고 보관합니다.
        for track_index in unmatched_tracks:
            self.tracks[track_index].mark_missed()

        # 4. 허용한 기간을 초과한 트랙을 제거합니다.
        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames <= self.max_missed
        ]

        # 5. 기존 트랙에 연결되지 않은 탐지는 새 트랙으로 등록합니다.
        for detection_index in unmatched_detections:
            new_track = Track(
                track_id=self.next_track_id,
                bbox=detection_boxes[detection_index],
            )

            self.tracks.append(new_track)
            self.next_track_id += 1

        # 이번 프레임에서 탐지와 연결됐거나 새로 생성된 트랙만 반환합니다.
        # 보관 중인 미연결 트랙은 내부 목록에는 남아 있습니다.
        return [
            track
            for track in self.tracks
            if track.missed_frames == 0
        ]

class TwoStageTracker(HighConfidenceTracker):
    """기존 추적기에 낮은 신뢰도 탐지를 사용하는 2차 매칭을 추가합니다."""

    # 부모 클래스의 __init__을 그대로 사용합니다.
    # 따라서 설정값, tracks, next_track_id도 동일하게 초기화됩니다.

    def update(self, frame_detections):
        high_detections, low_detections = split_detections(
            frame_detections,
            high_threshold=self.high_threshold,
            low_threshold=0.1,
        )

        high_boxes = [
            detection["bbox"]
            for detection in high_detections
        ]

        low_boxes = [
            detection["bbox"]
            for detection in low_detections
        ]

        # 1. 기존 트랙의 현재 위치를 예측합니다.
        for track in self.tracks:
            track.predict()

        predicted_boxes = [
            track.bbox
            for track in self.tracks
        ]

        # 2. 높은 신뢰도 탐지와 먼저 연결합니다.
        matches, unmatched_tracks, unmatched_high = match_boxes(
            predicted_boxes,
            high_boxes,
            min_iou=self.min_iou,
        )

        for track_index, detection_index in matches:
            self.tracks[track_index].update(
                high_boxes[detection_index]
            )

        # 3. 2차 매칭 후보를 고릅니다.
        # 1차에서 연결되지 않았고, 직전 프레임에는 관측된 트랙입니다.
        candidate_indices = [
            track_index
            for track_index in unmatched_tracks
            if self.tracks[track_index].missed_frames == 0
        ]

        candidate_boxes = [
            self.tracks[track_index].bbox
            for track_index in candidate_indices
        ]

        low_matches, _, _ = match_boxes(
            candidate_boxes,
            low_boxes,
            min_iou=self.min_iou,
        )

        recovered_tracks = set()

        for candidate_index, detection_index in low_matches:
            # candidate_index는 후보 목록 안의 인덱스입니다.
            # 원래 self.tracks에서의 인덱스로 변환합니다.
            track_index = candidate_indices[candidate_index]

            self.tracks[track_index].update(
                low_boxes[detection_index]
            )

            recovered_tracks.add(track_index)

        # 4. 두 번의 매칭에서도 연결되지 않은 트랙만 미연결 처리합니다.
        for track_index in unmatched_tracks:
            if track_index not in recovered_tracks:
                self.tracks[track_index].mark_missed()

        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames <= self.max_missed
        ]

        # 5. 연결되지 않은 높은 신뢰도 탐지만 새 트랙으로 만듭니다.
        for detection_index in unmatched_high:
            self.tracks.append(
                Track(
                    track_id=self.next_track_id,
                    bbox=high_boxes[detection_index],
                )
            )

            self.next_track_id += 1

        return [
            track
            for track in self.tracks
            if track.missed_frames == 0
        ]

def main():
    parser = argparse.ArgumentParser(
        description="탐지 CSV를 읽고 프레임별 결과를 확인합니다."
    )

    parser.add_argument(
        "--detections-csv",
        type=Path,
        required=True,
        help="detect_video.py가 저장한 detections.csv 경로",
    )

    parser.add_argument(
        "--frame-id",
        type=int,
        default=55,
        help="확인할 프레임 번호",
    )

    args = parser.parse_args()

    if args.frame_id < 1:
        parser.error("--frame-id는 1 이상이어야 합니다.")

    # CSV 전체를 읽고 프레임별로 묶습니다.
    detections = load_detections(args.detections_csv)

    # 각 프레임에 들어 있는 탐지 개수를 모두 더합니다.
    total_detections = sum(
        len(frame_detections)
        for frame_detections in detections.values()
    )

    # 해당 프레임에 탐지가 없으면 빈 리스트를 반환합니다.
    selected = detections.get(args.frame_id, [])

    # 선택한 프레임의 탐지를 높은 점수와 낮은 점수로 나눕니다.
    high_detections, low_detections = split_detections(selected)

    print(f"높은 신뢰도 탐지 수: {len(high_detections)}")
    print(f"낮은 신뢰도 탐지 수: {len(low_detections)}")

    print(f"전체 탐지 수: {total_detections}")
    print(f"탐지가 있는 프레임 수: {len(detections)}")
    print(f"선택한 프레임: {args.frame_id}")
    print(f"선택한 프레임의 탐지 수: {len(selected)}")

    for index, detection in enumerate(selected, start=1):
        print(f"{index}: {detection}")
        
    # 탐지 CSV와 같은 폴더에 있는 실행 기록을 읽습니다.
    run_path = args.detections_csv.parent / "run.json"

    with run_path.open(encoding="utf-8") as file:
        run_info = json.load(file)

    # 탐지가 없는 프레임도 처리하기 위해 전체 프레임 수를 사용합니다.
    total_frames = run_info["processed_frames"]

    # 기존 HighConfidenceTracker 대신 사용합니다.
    tracker = TwoStageTracker(
        high_threshold=0.6,
        min_iou=0.3,
        max_missed=20,
    )

    # 기존 기준 결과를 보존하도록 다른 이름으로 저장합니다.
    output_path = args.detections_csv.parent / "tracks_two_stage.csv"

    saved_rows = 0

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        # 탐지 CSV와 달리, 차량을 구분하는 track_id가 들어갑니다.
        writer.writerow([
            "frame_id",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
        ])

        print("\n전체 영상 추적 시작")

        for frame_id in range(1, total_frames + 1):
            frame_detections = detections.get(frame_id, [])

            visible_tracks = tracker.update(frame_detections)

            # 현재 관측된 트랙 하나를 CSV 한 줄로 저장합니다.
            for track in visible_tracks:
                writer.writerow([
                    frame_id,
                    track.track_id,
                    *track.bbox,
                ])

                saved_rows += 1

            if frame_id % 100 == 0 or frame_id == total_frames:
                visible_ids = [
                    track.track_id
                    for track in visible_tracks
                ]

                print(
                    f"프레임 {frame_id}/{total_frames}, "
                    f"현재 관측된 ID: {visible_ids}"
                )

    print("추적 완료")
    print(f"처리 프레임 수: {total_frames}")
    print(f"누적 발급 ID 수: {tracker.next_track_id - 1}")
    print(f"저장된 추적 행 수: {saved_rows}")
    print(f"추적 CSV: {output_path.resolve()}")

if __name__ == "__main__":
    main()