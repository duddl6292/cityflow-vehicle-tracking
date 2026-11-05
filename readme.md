# CityFlow Vehicle Tracking

**논문 핵심 구현 기반 차량 추적·카메라 간 검색 프로젝트** · 진행 중

CityFlowV2 CCTV 영상에서 차량을 추적하고, 다른 카메라의 동일 차량을 검색하는 프로젝트입니다. 현재는 YOLO11s의 고정된 검출 결과를 입력으로 ByteTrack의 2단계 매칭 핵심을 구현하고, 단일 카메라에서 비교 실험과 실패 분석을 진행했습니다.

> 업데이트: 2026-11-05 · 현재 검증 범위: `train/S01/c001`, 1,955프레임 · GT 기반 정량 평가 진행 예정

## Problem & Goal

- **추적:** 낮은 신뢰도 탐지를 활용한 2차 매칭이 차량 ID 유지에 도움이 되는가?
- **검색:** 여러 프레임의 특징을 결합하면 단일 프레임보다 동일 차량을 잘 찾는가?
- **구조 구현:** 직접 작성한 MS-Block이 공식 구현과 같은 조건에서 일치하는가?

## Design & Scope

| 단계 | 구성 | 직접 구현·실험할 부분 | 상태 |
| --- | --- | --- | --- |
| 준비 | YOLO11s | 영상·GT 처리, 박스·신뢰도 저장 | 단일 카메라 실행 확인 |
| 1. Tracking | YOLO11s + ByteTrack 핵심 | Kalman Filter, 2단계 매칭, ID·트랙 관리 | 구현·첫 비교 완료, 정량 평가 예정 |
| 2. Retrieval | 위 구성 + TransReID | 차량 이미지 수집, 특징 결합, 카메라 간 검색 | 예정 |
| 3. Detection Block | YOLO-MS | MS-Block 부분 구현, 공식 구현과 구조·출력 비교 | 예정 |

처리 흐름: 차량 검출 → 카메라 내부 추적 ID 생성 → 차량 이미지 특징 추출 → 다른 카메라의 유사 차량 검색.

**직접 작성:** CSV·GT 파싱, IoU·비용 행렬, 매칭 순서와 후보 관리, NumPy 기반 Kalman Filter 초기화·예측·보정, ID 생성·미연결 보관·제거, 결과 저장·시각화.

**외부 활용:** Ultralytics YOLO11s COCO 사전학습 모델, OpenCV 영상 처리, SciPy `linear_sum_assignment`의 선형 할당 계산. 현재 단계에서 검출 모델은 재학습하지 않았습니다.

현재 추적기는 ByteTrack의 핵심 아이디어를 검증하는 구현입니다. 확정 대기 트랙·중복 트랙 정리·신뢰도 융합·매칭 세부 설정 등에서 공식 구현과 차이가 있으며, 전체 시스템의 성능 재현을 의미하지 않습니다.

## First Experiment

### 고정 조건

| 항목 | 설정 |
| --- | --- |
| 영상 | `train/S01/c001` · 1920×1080 · 10 FPS · 1,955프레임 |
| 검출기 | `yolo11s.pt` · COCO 사전학습 가중치 |
| 검출 설정 | `imgsz=640`, `rect=False`, `conf=0.1`, NMS `iou=0.7` |
| 차량 클래스 | COCO `car=2`, `bus=5`, `truck=7` |
| 추적 설정 | 높은 신뢰도 `≥0.6`, 낮은 신뢰도 `0.1~0.6 미만`, 매칭 IoU `≥0.3` |
| 미연결 트랙 보관 | 최대 20프레임, 이 영상에서 약 2초 |
| 비교 방법 | 동일한 탐지 CSV에서 2차 매칭만 추가 |

검출 결과 23,099개를 저장하고, 두 추적 방식 모두 1,955개 프레임을 처리했습니다.

| 관찰 결과 | 높은 신뢰도 탐지만 사용 | 2차 매칭 추가 |
| --- | ---: | ---: |
| 누적 발급 추적 ID | 184 | 158 |

**누적 발급 ID는 실제 차량 수나 정확도 지표가 아닙니다.** ID 감소만으로 성능 개선을 확정하지 않으며, IDF1·GT 기준 ID 변경·궤적 단절 평가는 아직 수행하지 않았습니다.

### 실패 사례: 70~150번 프레임의 대형 트럭

- **주차 위치의 차량:** 2차 매칭으로 ID 4를 95번까지 유지했습니다. 96번에는 예측 박스와 모든 탐지의 최대 IoU가 약 0.242로 기준에 미달했습니다. 이후 예측 위치 변화와 재매칭 조건으로 복구되지 않았고, 116번에서 보관 기간 초과로 제거된 뒤 129번에 ID 18이 생성됐습니다.
- **대형 트럭:** 113번에는 앞부분·뒤쪽·전체로 해석되는 탐지 박스가 동시에 존재했습니다. 앞부분은 ID 10에 연결되고 뒤쪽은 ID 15로 생성됐으며, 이후 넓은 전체 박스가 ID 15에 연결됐습니다. 검출 박스의 분할·변동과 추적 ID의 분리를 함께 확인한 사례입니다.

## Dataset & Evaluation

**CityFlowV2 — AI City Challenge 2022, Track 1**을 사용합니다. 현재 `train/S01/c001`은 구현 점검용입니다. 공식 데이터 분할을 유지하고, 비교용 validation 카메라를 별도로 선정할 예정입니다.

GT는 카메라 두 대 이상을 통과한 차량만 포함합니다. 따라서 GT와 연결되지 않은 예측을 모두 오탐으로 계산하지 않고, 평가 대상과 공식 도구의 필터링·제외 규칙을 먼저 확인합니다.

| 대상 | 예정된 검증 |
| --- | --- |
| 추적 | 같은 검출 입력에서 2차 매칭 유무 비교: IDF1, ID 변경, 궤적 단절 |
| 검색 | 고정된 query/gallery에서 단일·여러 프레임 특징 비교: Rank-1, 검색 mAP |
| MS-Block | 같은 구조·가중치·입력·평가 모드에서 출력 형태와 수치 오차 비교 |

카메라마다 추적 ID는 독립적입니다. 여러 카메라의 같은 번호가 동일 차량을 의미하지 않으며, 카메라 간 검색은 후속 단계에서 구현합니다.

## Run

프로젝트 루트의 Windows PowerShell 기준입니다. Python 환경에는 PyTorch, torchvision, Ultralytics, OpenCV, NumPy, SciPy가 필요합니다.

| 확인한 환경 | 버전·장치 |
| --- | --- |
| Python | 3.12.10 |
| PyTorch / torchvision | 2.9.1+cu128 / 0.24.1+cu128 |
| Ultralytics / OpenCV | 8.4.202 / 5.0.0 |
| GPU | NVIDIA GeForce RTX 4060 Ti 8GB |

### 영상·GT 확인과 검출 결과 생성

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_video.py --camera-dir .\dataset\AICity22_Track1_MTMC_Tracking\train\S01\c001
.\.venv\Scripts\python.exe .\scripts\visualize_gt.py --camera-dir .\dataset\AICity22_Track1_MTMC_Tracking\train\S01\c001 --frame-id 55
.\.venv\Scripts\python.exe .\scripts\detect_frame.py --camera-dir .\dataset\AICity22_Track1_MTMC_Tracking\train\S01\c001 --frame-id 55 --imgsz 640 --conf 0.1
.\.venv\Scripts\python.exe .\scripts\detect_video.py --camera-dir .\dataset\AICity22_Track1_MTMC_Tracking\train\S01\c001 --imgsz 640 --conf 0.1
```

전체 영상 검출 결과는 `outputs/detections/<실행폴더>/`에 `detections.csv`, `detections.mp4`, `run.json`으로 저장됩니다. 아래 경로는 이번 비교 실험의 실행폴더이며, 새로 검출했다면 해당 폴더로 바꿉니다.

```powershell
$runDir = ".\outputs\detections\S01_c001_20261104_210010_620005"
.\.venv\Scripts\python.exe .\scripts\inspect_detections.py --detections-csv "$runDir\detections.csv" --frame-id 1955
```

현재 `--frame-id`는 선택 프레임의 탐지 출력에 사용되며, 추적은 `run.json`의 `processed_frames`를 기준으로 전체 프레임에 수행됩니다.

### 두 추적 방식 선택

현재는 `inspect_detections.py`의 `main()`에서 추적기와 출력 파일명을 바꿔 각각 실행합니다. 추적기 선택 CLI는 아직 분리하지 않았습니다.

```python
# 높은 신뢰도 탐지만 사용
tracker = HighConfidenceTracker(high_threshold=0.6, min_iou=0.3, max_missed=20)
output_path = args.detections_csv.parent / "tracks_high_only.csv"

# 2차 매칭 추가: 위 두 줄 대신 사용
tracker = TwoStageTracker(high_threshold=0.6, min_iou=0.3, max_missed=20)
output_path = args.detections_csv.parent / "tracks_two_stage.csv"
```

추적 CSV는 `frame_id,track_id,x1,y1,x2,y2` 형식입니다. 현재 관측된 트랙만 행으로 저장하며, 출력 행이 없는 프레임도 내부 추적 처리에는 포함됩니다.

```powershell
.\.venv\Scripts\python.exe .\scripts\visualize_tracks.py --tracks-csv "$runDir\tracks_high_only.csv"
.\.venv\Scripts\python.exe .\scripts\visualize_tracks.py --tracks-csv "$runDir\tracks_two_stage.csv"
```

각 CSV와 같은 폴더에 같은 이름의 MP4가 생성됩니다. 원본 데이터·가중치·생성 영상은 저장소에 포함하지 않습니다.

## Progress & Next

- [x] 데이터 구조·영상 정보 확인, 지정 프레임 GT 시각화
- [x] YOLO11s 단일 프레임·전체 영상 추론, CSV·영상·환경 기록 저장
- [x] IoU·선형 할당 연결, Kalman Filter·트랙 ID 관리
- [x] 높은 신뢰도만 사용하는 기준 추적기와 2차 매칭 버전 비교
- [x] 추적 영상 확인, 대형 트럭·가림 구간의 첫 실패 분석
- [ ] 공식 ByteTrack과 구현 차이 정리, 재사용 코드·실행 스크립트 분리
- [ ] GT 평가 대상·제외 규칙 확정, 정량 비교와 검증 카메라 확장
- [ ] TransReID 기반 차량 검색·특징 결합 실험
- [ ] YOLO-MS MS-Block 구현·검증

## References

- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864) · [Official code](https://github.com/FoundationVision/ByteTrack) — 핵심 추적 로직 구현 참고.
- [TransReID: Transformer-Based Object Re-Identification](https://arxiv.org/abs/2102.04378) · [Official code](https://github.com/damo-cv/TransReID) — 후속 검색 실험.
- [YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection](https://arxiv.org/abs/2308.05480) · [Official code](https://github.com/FishAndWasabi/YOLO-MS) — 후속 MS-Block 구현.
- [Multi-Camera Vehicle Tracking System for AI City Challenge 2022](https://openaccess.thecvf.com/content/CVPR2022W/AICity/html/Li_Multi-Camera_Vehicle_Tracking_System_for_AI_City_Challenge_2022_CVPRW_2022_paper.html) — 시스템 구성 참고.
