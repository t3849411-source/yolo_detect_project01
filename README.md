# AI 화염·연기 탐지

스마트폰의 갤러리 사진 또는 카메라 프레임을 PC의 YOLOv8 모델로 분석하는 모바일 우선 웹 서비스입니다. Vue 3가 화면과 카메라를 담당하고, FastAPI가 `models/best.pt`를 서버 시작 시 한 번만 로딩하여 REST 및 WebSocket 추론을 제공합니다. CUDA가 가능하면 GPU를 사용하고, 그렇지 않으면 CPU로 전환합니다.

> 이 서비스는 학습 및 보조 탐지용이며 실제 화재경보기, 소방설비, 안전요원의 판단을 대체하지 않습니다. 실제 화재가 의심되면 모델 결과와 관계없이 즉시 안전 절차와 신고 체계를 따르세요.

## 시스템 구조

```text
스마트폰/PC 브라우저
  ├─ 갤러리 사진 ─ multipart/form-data ─┐
  └─ 카메라 프레임 ─ JPEG/WebP WebSocket ├─ FastAPI ─ YOLOv8n ─ CUDA/CPU
                                          └─ JSON 좌표 + 결과 JPEG
```

- 개발: Vue `0.0.0.0:5173` → Vite proxy → FastAPI `0.0.0.0:8000`
- production: `frontend/dist`를 FastAPI가 제공하여 UI, API, WebSocket이 같은 origin을 사용
- 실시간 전송: 이전 응답 전에는 다음 프레임을 보내지 않으며 서버도 연결별 최신 1프레임만 유지
- 좌표 표시: `object-fit: cover/contain`의 scale과 offset을 반영하여 Canvas 좌표 변환
- 경고: 화염은 빨간색, 연기는 주황색; 최근 탐지를 1.8초 유지하고 진동/소리는 사용자 opt-in 및 5초 cooldown 적용

## 데이터셋과 전처리 결과

결합한 원본은 다음 두 Kaggle 데이터셋입니다.

- `dataclusterlabs/fire-and-smoke-dataset`
- `azimjaan21/fire-and-smoke-dataset-object-detection-yolo`

두 번째 데이터셋의 `other` 클래스 박스는 제거하되 해당 이미지는 negative sample로 유지했습니다. 완료된 전처리 결과는 다음과 같습니다.

| 항목 | 결과 |
|---|---:|
| 입력 이미지 | 17,661 |
| 정확 중복 제거 | 715 |
| 최종 이미지 | 16,946 |
| train | 11,847 |
| validation | 3,395 |
| test | 1,704 |
| fire 객체 | 5,175 |
| smoke 객체 | 5,262 |
| 잘못된 박스 제거 | 1 |
| split 간 중복/유사 프레임 누수 | 0 |

최종 클래스는 `0: fire`, `1: smoke`입니다. raw 원본과 학습 split은 독립 복사본이며 hardlink가 아닙니다. 현재 데이터셋은 완료 상태이므로 서비스를 실행하기 위해 전처리를 다시 할 필요가 없습니다.

## 학습 설정과 성능

- 모델: YOLOv8n
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB
- 입력 크기: 640
- batch: 64
- epochs: 50
- seed: 42
- weights: `models/best.pt`, `models/last.pt`
- 학습 결과: `outputs/training/fire-smoke-yolov8n`

### Validation과 독립 test 비교

독립 test 평가는 `data/fire.yaml`의 `test: images/test` 1,704장만 사용했습니다.

| 지표 | Validation | Test | Test - Validation |
|---|---:|---:|---:|
| Precision | 0.557 | 0.486 | -0.071 |
| Recall | 0.540 | 0.642 | +0.102 |
| mAP50 | 0.524 | 0.555 | +0.031 |
| mAP50-95 | 0.242 | 0.278 | +0.036 |
| fire mAP50-95 | 0.223 | 0.289 | +0.066 |
| smoke mAP50-95 | 0.260 | 0.267 | +0.007 |

### Test 클래스별 성능

| 클래스 | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| fire | 0.485 | 0.576 | 0.573 | 0.289 |
| smoke | 0.486 | 0.708 | 0.537 | 0.267 |

평가 원본은 [evaluation_report.md](outputs/evaluation/evaluation_report.md), [metrics.json](outputs/evaluation/metrics.json), [validation_vs_test.csv](outputs/evaluation/validation_vs_test.csv)에 있습니다. FP/FN 예시는 confidence 0.25, 같은 클래스, IoU 0.50 기준으로 실제 정답과 예측을 매칭해 선정했습니다.

### 평가 그래프

| PR Curve | F1 Curve |
|---|---|
| ![PR Curve](outputs/evaluation/metrics/BoxPR_curve.png) | ![F1 Curve](outputs/evaluation/metrics/BoxF1_curve.png) |

| Precision Curve | Recall Curve |
|---|---|
| ![Precision Curve](outputs/evaluation/metrics/BoxP_curve.png) | ![Recall Curve](outputs/evaluation/metrics/BoxR_curve.png) |

| Confusion Matrix | Normalized Confusion Matrix |
|---|---|
| ![Confusion Matrix](outputs/evaluation/metrics/confusion_matrix.png) | ![Normalized Confusion Matrix](outputs/evaluation/metrics/confusion_matrix_normalized.png) |

추가 예시는 `outputs/evaluation/samples`, `false_positives`, `false_negatives`, `undetected_negatives`에 있습니다.

## 설치

WSL에서 프로젝트 루트로 이동합니다.

```bash
cd /mnt/c/Users/SSAFY/Desktop/work/fire-detection-yolov8
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cd frontend
npm ci
cd ..
```

GPU 확인:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 개발 서버 실행

터미널 1, FastAPI:

```bash
cd /mnt/c/Users/SSAFY/Desktop/work/fire-detection-yolov8
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

터미널 2, Vue:

```bash
cd /mnt/c/Users/SSAFY/Desktop/work/fire-detection-yolov8/frontend
npm run dev -- --host 0.0.0.0
```

PC에서는 `http://localhost:5173`을 엽니다. Vite가 `/api`와 `/ws`를 8000번 포트로 proxy합니다. 주소는 코드에 하드코딩되어 있지 않습니다.

## Production 실행

```bash
cd /mnt/c/Users/SSAFY/Desktop/work/fire-detection-yolov8/frontend
npm run build
cd ..
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

이제 `http://localhost:8000/`에서 UI가 제공됩니다.

```text
https://접속주소/
https://접속주소/api/health
wss://접속주소/ws/detect
```

페이지가 HTTPS이면 프론트가 자동으로 `wss://`, HTTP이면 `ws://`를 선택합니다.

## Render 배포

이 저장소는 루트의 `render.yaml`과 `Dockerfile`로 Render Blueprint 배포를 지원합니다. Docker 빌드 단계에서 Vue production bundle을 생성하고, 실행 단계에서는 CPU용 PyTorch와 FastAPI를 설치합니다. 배포된 한 주소에서 웹 화면, REST API, WebSocket을 모두 제공합니다.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2Ft3849411-source%2Fyolo_detect_project01)

수동으로 만들 때는 Render Dashboard에서 **New → Blueprint**를 선택하고 이 GitHub 저장소를 연결합니다. `render.yaml`이 다음 설정을 자동 적용합니다.

- 서비스 유형: Docker Web Service
- 리전: Singapore
- health check: `/api/health`
- 모델: 저장소에 포함된 `models/best.pt`
- 브라우저 접속: Render가 발급한 `https://...onrender.com/`
- WebSocket: 화면이 현재 origin을 기준으로 자동으로 `wss://.../ws/detect` 사용

무료 인스턴스는 GPU가 없으므로 CPU 추론을 사용하며, 유휴 상태에서 sleep 후 첫 요청이 느릴 수 있습니다. 결과 이미지는 Render 인스턴스의 임시 파일시스템에 저장되므로 재배포나 재시작 후 유지된다고 가정하지 마세요. 공개 URL은 누구나 모델을 호출할 수 있으므로 실제 운영 시 인증, 요청 제한, 모니터링을 추가하는 것이 좋습니다.

## 사용 방법

### 갤러리 사진 탐지

1. 첫 화면에서 **갤러리 사진 탐지**를 선택합니다.
2. JPEG, PNG, WebP 사진을 선택합니다. 파일 입력에 `capture` 속성을 사용하지 않으므로 카메라 촬영을 강제하지 않습니다.
3. 필요하면 Confidence와 IoU를 조정하고 **탐지 시작**을 누릅니다.
4. 원본/결과, 클래스별 confidence, 추론 시간과 탐지 개수를 확인합니다.
5. **결과 이미지 다운로드** 또는 **다른 사진 선택**을 사용합니다.

### 실시간 카메라 탐지

1. 첫 화면에서 **카메라 실시간 탐지**를 선택합니다.
2. **카메라 시작**을 누르고 권한을 허용합니다. 후면 카메라를 우선 요청합니다.
3. 전송 FPS(1/3/5/10, 기본 5)와 Confidence를 조정합니다.
4. 전송 프레임은 640×360, JPEG quality 0.7입니다.
5. 필요하면 **전면·후면 전환**, **카메라 중지**를 사용합니다.
6. 진동은 체크박스를 직접 켠 경우에만, 소리는 **소리 경고 활성화**를 누른 경우에만 동작합니다.

백그라운드 탭에서는 프레임 전송을 멈추고 복귀 시 재개합니다. 중지·뒤로가기·컴포넌트 제거 시 camera track, timer, WebSocket, AudioContext를 정리합니다.

## 스마트폰 접속, WSL, 방화벽

### PC 로컬 IP 확인

Windows PowerShell에서 다음 명령으로 Wi-Fi/Ethernet의 IPv4 주소를 확인합니다.

```powershell
ipconfig
```

PC와 스마트폰을 같은 사설 네트워크에 연결하고 production 서버를 실행한 뒤 `https://<PC_IP>:8000/`으로 접속합니다. 카메라는 secure context가 필요하므로 스마트폰에서 단순 `http://<PC_IP>`만으로 동작한다고 가정하면 안 됩니다. 갤러리 업로드는 HTTP에서도 가능하지만 HTTPS 사용을 권장합니다.

### WSL 외부 접속

Windows 11의 WSL mirrored networking을 권장합니다. `%UserProfile%\.wslconfig` 예시:

```ini
[wsl2]
networkingMode=mirrored
```

설정 후 PowerShell에서 `wsl --shutdown`하고 WSL을 다시 시작합니다. NAT 모드에서 직접 접근이 안 되면 WSL IP를 확인하고 관리자 PowerShell에서 portproxy를 설정할 수 있습니다.

```powershell
wsl hostname -I
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=<WSL_IP> connectport=8000
```

WSL IP는 재시작 후 달라질 수 있습니다. 필요 없어진 규칙은 삭제합니다.

```powershell
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8000
```

### Windows 방화벽

관리자 PowerShell에서 사설 네트워크에만 포트를 허용합니다.

```powershell
New-NetFirewallRule -DisplayName "Fire Detection 8000 Private" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
New-NetFirewallRule -DisplayName "Fire Detection 5173 Private" -Direction Inbound -Protocol TCP -LocalPort 5173 -Action Allow -Profile Private
```

production만 사용하면 5173 규칙은 필요 없습니다. 작업 후 `Remove-NetFirewallRule -DisplayName "..."`으로 제거할 수 있습니다.

## 로컬 HTTPS와 HTTPS 터널

### mkcert 로컬 인증서

개발용 PC와 본인이 관리하는 스마트폰에서만 사용하세요.

```powershell
mkcert -install
New-Item -ItemType Directory -Force certs
mkcert -cert-file certs/dev.pem -key-file certs/dev-key.pem <PC_IP> localhost 127.0.0.1
```

production build 후 WSL에서:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 \
  --ssl-certfile certs/dev.pem \
  --ssl-keyfile certs/dev-key.pem
```

`mkcert -CAROOT`로 확인한 로컬 CA를 스마트폰에 설치하고 신뢰해야 합니다. 인증서 개인키와 root CA를 저장소에 커밋하지 마세요.

### HTTPS 터널

외부 공개가 허용된 경우에만 다음 중 하나를 사용합니다. 이 프로젝트 작업에서는 터널을 자동 생성하거나 공개하지 않았습니다.

```bash
cloudflared tunnel --url http://localhost:8000
# 또는
ngrok http 8000
```

발급된 HTTPS 주소로 접속하면 프론트는 같은 host의 `wss://.../ws/detect`를 사용합니다. 공개 전 접근 통제, 로그, 조직 정책을 확인하세요.

## API 명세

| 메서드/경로 | 설명 |
|---|---|
| `GET /api/health` | 서버 상태, 모델 준비 여부, 장치 |
| `GET /api/model-info` | 모델 경로/이름, 클래스, 장치, 로딩 오류 |
| `POST /api/detect/image` | `file`, `confidence`, `iou` multipart 업로드 |
| `GET /api/results/{filename}` | UUID 결과 JPEG 다운로드 |
| `WS /ws/detect?confidence=.35&iou=.45` | Binary JPEG/WebP 프레임 → 좌표 JSON |

이미지 API는 MIME과 확장자, 빈 파일, 손상 이미지, 기본 15MB 제한, 최대 5천만 pixel, Confidence/IoU 범위를 검증합니다. 결과 파일은 `outputs/results`에 저장되고 기본 24시간 뒤 서버 시작 또는 새 결과 생성 시 정리됩니다.

주요 환경 변수:

| 변수 | 기본값 |
|---|---|
| `MODEL_PATH` | `models/best.pt` |
| `RESULTS_DIR` | `outputs/results` |
| `MAX_UPLOAD_MB` | `15` |
| `MAX_WS_FRAME_MB` | `8` |
| `MAX_IMAGE_PIXELS` | `50000000` |
| `RESULT_RETENTION_HOURS` | `24` |
| `DEFAULT_CONFIDENCE` | `0.35` |
| `DEFAULT_IOU` | `0.45` |

## 테스트

```bash
cd /mnt/c/Users/SSAFY/Desktop/work/fire-detection-yolov8
source .venv/bin/activate
pytest -q
python training/model_smoke_test.py --model models/best.pt --output outputs/model_smoke_test
cd frontend
npm test
npm run build
```

현재 검증 결과:

- 백엔드: 17 passed
- 프론트엔드: 12 passed
- production build: 성공
- 실제 CUDA smoke: fire, smoke, fire+smoke 합성 입력, negative, 3024×4032, EXIF Orientation 입력 모두 통과
- 실제 REST/WS: `cuda:0`, 정상 JSON/좌표/결과 JPEG 확인

### 실제 스마트폰 수동 체크리스트

- [ ] HTTPS 주소에서 카메라 권한 허용/거부 문구가 올바르다.
- [ ] iOS Safari와 Android Chrome에서 후면 카메라가 우선 열린다.
- [ ] 전면·후면 전환 시 이전 track이 종료된다.
- [ ] 세로·가로 회전과 `object-fit: cover`에서 박스가 물체와 일치한다.
- [ ] 탭을 백그라운드로 보내면 전송이 멈추고 복귀하면 재개된다.
- [ ] 느린 Wi-Fi에서도 이전 응답 전 다음 프레임을 보내지 않는다.
- [ ] WebSocket 단절 후 최대 3회 재연결하고 사용자가 중지하면 재연결하지 않는다.
- [ ] 경고가 한 프레임 누락으로 깜빡이지 않는다.
- [ ] 진동/소리가 opt-in 이후에만 cooldown을 두고 동작한다.
- [ ] EXIF 세로 사진과 결과 이미지 방향/박스가 일치한다.
- [ ] 결과 이미지 다운로드가 동작한다.

## 오류 해결

- **모델 파일이 없습니다 / 503**: `models/best.pt` 존재 여부와 `MODEL_PATH`를 확인하고 서버를 재시작합니다.
- **CUDA 대신 CPU**: `nvidia-smi`와 `torch.cuda.is_available()`을 확인합니다. Windows NVIDIA 드라이버와 WSL GPU 연동을 점검합니다.
- **카메라 권한 오류**: HTTPS 여부, 브라우저 사이트 권한, OS 카메라 권한을 확인합니다.
- **스마트폰 접속 불가**: 같은 네트워크인지, `0.0.0.0` 바인딩인지, Windows 방화벽과 WSL mirrored/NAT 설정을 확인합니다.
- **413**: 15MB 이하 이미지를 사용하거나 운영 정책에 맞게 `MAX_UPLOAD_MB`를 변경합니다.
- **손상된 이미지**: 다른 앱에서 다시 저장하거나 JPEG/PNG/WebP로 변환합니다.
- **WebSocket 연결 실패**: HTTPS 페이지에서는 반드시 WSS가 필요합니다. reverse proxy가 WebSocket upgrade를 전달하는지 확인합니다.
- **탐지 결과 없음**: Confidence를 낮춰 비교하되, 실제 안전 판단에는 사용하지 마세요.

모델 재학습과 데이터 재전처리는 현재 서비스 완성 및 실행에 필요하지 않습니다.
