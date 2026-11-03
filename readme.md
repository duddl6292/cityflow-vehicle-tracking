CityFlow Vehicle Tracking
논문 핵심 구현 기반 차량 추적·카메라 간 검색 프로젝트 · 진행 중

CityFlowV2 CCTV 영상에서 차량을 추적하고, 다른 카메라에 등장한 동일 차량을 검색합니다. ByteTrack의 연관 알고리즘을 직접 구현하고, TransReID 특징 결합 실험과 YOLO-MS의 MS-Block 구현으로 확장합니다.
Problem & Goal
가림으로 탐지 신뢰도가 낮아지면 차량 추적이 끊길 수 있고, 카메라에 따른 외형 변화는 동일 차량 검색을 어렵게 합니다. 다음 질문을 구현과 실험으로 검증합니다.
- 추적: 저신뢰도 박스를 활용하는 2차 매칭이 차량 ID 유지에 도움이 되는가?
- 검색: 여러 프레임의 특징을 결합하면 한 프레임보다 동일 차량을 잘 찾는가?
- 구조 구현: 직접 작성한 MS-Block이 공식 구현과 같은 조건에서 일치하는가?
Design & Scope
단계	구성	직접 구현·실험할 부분
준비	YOLO11s	영상 입력, 검출 박스·신뢰도 저장
1. Tracking	YOLO11s + ByteTrack	2단계 매칭·트랙 상태 관리, 2차 매칭 유무 비교
2. Retrieval	YOLO11s + ByteTrack + TransReID	차량 이미지 수집·특징 결합·카메라 간 검색
3. Detection Block	YOLO-MS	MS-Block 부분 구현, 공식 구현과 출력 비교


처리 흐름: 영상의 차량 검출 → 카메라 내부 ID·궤적 생성 → 차량 이미지 특징 추출 → 다른 카메라의 유사 차량 검색.

YOLO11s와 TransReID의 사전학습 모델은 활용하고, 표에 명시한 핵심 로직을 직접 작성합니다. 처음에는 카메라 하나에서 시작해 검색 단계에서 2~3개로 확장합니다. YOLO-MS 전체 탐지기 통합과 시간·카메라 관계를 활용한 자동 궤적 연결은 검증 후 확장합니다.
Evaluation
대상	비교·검증
추적	같은 검출 입력에서 2차 매칭 유무 비교: IDF1, ID 변경, 궤적 단절
검색	고정된 조회/후보 구성에서 단일/여러 프레임 비교: Rank-1, 검색 mAP
MS-Block	같은 구조·가중치·입력·평가 모드에서 출력 형태와 수치 오차 비교


공식 데이터 분할을 유지하고 비교 데이터와 설정을 고정합니다. 구조 검증과 전체 모델의 성능 재현을 구분하며, 성능 결과는 실험 완료 후 추가합니다.
Dataset
CityFlowV2 — AI City Challenge 2022, Track 1을 사용합니다. 카메라 두 대 이상을 통과한 차량만 GT에 포함되므로, 평가 시 공식 도구의 필터링·제외 규칙을 확인합니다. 원본 데이터는 저장소에 포함하지 않습니다.
Progress
- [x] 데이터 구조 확인 및 영상 정보 조회
- [x] 지정 프레임의 GT 시각화·샘플 확인
- [ ] YOLO11s 검출 입력 준비
- [ ] ByteTrack 핵심 구현·비교 실험
- [ ] TransReID 기반 차량 검색·특징 결합 실험
- [ ] YOLO-MS MS-Block 구현·검증
Run — 현재 구현된 기능
OpenCV가 설치된 Python 환경에서 프로젝트 최상위 위치를 기준으로 실행합니다. 카메라 경로는 데이터 위치에 맞게 지정합니다.
python scripts/inspect_video.py --camera-dir dataset/AICity22_Track1_MTMC_Tracking/train/S01/c001
python scripts/visualize_gt.py --camera-dir dataset/AICity22_Track1_MTMC_Tracking/train/S01/c001 --frame-id 55
결과 이미지는 기본적으로 outputs/에 저장됩니다. 상세 구현 계획은 [project_plan.md](docs/project_plan.md)를 참고하세요.
References
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box · Official code — 핵심 추적 로직 구현.
- TransReID: Transformer-Based Object Re-Identification · Official code — 공식 모델 활용·특징 결합 검색 실험.
- YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection · Official code — MS-Block 부분 구현·검증.
- Multi-Camera Vehicle Tracking System for AI City Challenge 2022 — 시스템 구성 참고.