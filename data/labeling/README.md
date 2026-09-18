# 라벨링 작업 폴더

원본 이미지를 CVAT 또는 Label Studio에 가져오고 `fire`, `smoke` 두 클래스로 실제 물체의 경계를 직접 표시합니다. 이미지 전체를 임의 박스로 지정하지 않습니다. 검수한 결과를 Pascal VOC, COCO 또는 YOLO 형식으로 `data/raw` 아래에 내보낸 뒤 `training/inspect_dataset.py`와 `training/prepare_dataset.py`를 실행합니다.
