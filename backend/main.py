from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, settings
from .detector import (
    Detector,
    DetectorUnavailableError,
    InvalidImageError,
    count_classes,
    draw_detections,
)


logger = logging.getLogger("uvicorn.error")
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp"}


def validate_threshold(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=422, detail=f"{name} 값은 0과 1 사이여야 합니다.")
    return value


def cleanup_old_results(directory: Path, retention_hours: int) -> int:
    if retention_hours <= 0 or not directory.is_dir():
        return 0
    cutoff = time.time() - retention_hours * 3600
    removed = 0
    for path in directory.glob("detection-*.jpg"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def create_app(app_settings: Settings = settings, detector: Detector | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("화염·연기 탐지 서버를 시작합니다.")
        app_settings.results_dir.mkdir(parents=True, exist_ok=True)
        removed = cleanup_old_results(app_settings.results_dir, app_settings.result_retention_hours)
        if removed:
            logger.info("만료된 결과 이미지 %d개를 정리했습니다.", removed)
        active_detector = detector or Detector(
            app_settings.model_path,
            max_image_pixels=app_settings.max_image_pixels,
        )
        if detector is None:
            await asyncio.to_thread(active_detector.load)
        app.state.detector = active_detector
        if active_detector.ready:
            logger.info(
                "모델 로딩 완료: %s (%s)",
                getattr(active_detector, "model_path", app_settings.model_path),
                active_detector.device,
            )
        else:
            logger.error("모델을 사용할 수 없습니다: %s", active_detector.error)
        try:
            yield
        finally:
            logger.info("화염·연기 탐지 서버를 종료합니다.")

    app = FastAPI(title="AI 화염·연기 탐지 API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=app_settings.allowed_origins != ("*",),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health(request: Request):
        current = request.app.state.detector
        return {
            "status": "ok" if current.ready else "degraded",
            "model_ready": current.ready,
            "device": current.device,
        }

    @app.get("/api/model-info")
    async def model_info(request: Request):
        return request.app.state.detector.info()

    @app.get("/api/results/{filename}", name="result_image")
    async def result_image(filename: str):
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.startswith("detection-") or not safe_name.endswith(".jpg"):
            raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")
        path = app_settings.results_dir / safe_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")
        return FileResponse(
            path,
            media_type="image/jpeg",
            filename=safe_name,
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/detect/image")
    async def detect_image(
        request: Request,
        file: UploadFile = File(...),
        confidence: float = Form(app_settings.default_confidence),
        iou: float = Form(app_settings.default_iou),
    ):
        validate_threshold("Confidence", confidence)
        validate_threshold("IoU", iou)
        suffix = Path(file.filename or "").suffix.lower()
        content_type = (file.content_type or "").lower().split(";", 1)[0]
        if content_type not in ALLOWED_MIME or suffix not in ALLOWED_SUFFIX:
            await file.close()
            raise HTTPException(status_code=415, detail="JPEG, PNG, WebP 이미지만 업로드할 수 있습니다.")
        data = await file.read(app_settings.max_upload_bytes + 1)
        await file.close()
        if not data:
            raise HTTPException(status_code=400, detail="빈 파일은 업로드할 수 없습니다.")
        if len(data) > app_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="업로드 파일이 허용 크기를 초과했습니다.")
        try:
            prediction = await asyncio.to_thread(
                request.app.state.detector.predict,
                data,
                confidence,
                iou,
            )
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DetectorUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        rendered = draw_detections(prediction.image, prediction.detections)
        filename = f"detection-{uuid.uuid4().hex}.jpg"
        output_path = app_settings.results_dir / filename
        buffer = io.BytesIO()
        rendered.save(buffer, format="JPEG", quality=92, optimize=True)
        output_path.write_bytes(buffer.getvalue())
        cleanup_old_results(app_settings.results_dir, app_settings.result_retention_hours)
        fire_count, smoke_count = count_classes(prediction.detections)
        return {
            "detected": bool(prediction.detections),
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "image_width": prediction.width,
            "image_height": prediction.height,
            "inference_ms": round(prediction.inference_ms, 2),
            "device": prediction.device,
            "result_image_url": str(request.url_for("result_image", filename=filename).path),
            "detections": prediction.detections,
        }

    @app.websocket("/ws/detect")
    async def websocket_detect(websocket: WebSocket):
        await websocket.accept()
        try:
            confidence = float(websocket.query_params.get("confidence", app_settings.default_confidence))
            iou = float(websocket.query_params.get("iou", app_settings.default_iou))
            if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
                await websocket.send_json({"error": "Confidence와 IoU는 0과 1 사이여야 합니다."})
                await websocket.close(code=1008)
                return
        except ValueError:
            await websocket.send_json({"error": "잘못된 threshold 값입니다."})
            await websocket.close(code=1008)
            return

        # 연결마다 독립적인 크기 1 큐를 두고, 밀린 경우 가장 오래된 프레임을 버린다.
        frame_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=1)
        disconnected = asyncio.Event()
        send_lock = asyncio.Lock()
        frame_id = 0

        async def send_json(payload: dict) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def receive_frames() -> None:
            nonlocal frame_id
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    data = message.get("bytes")
                    if not data:
                        await send_json({"error": "Binary JPEG 또는 WebP 프레임이 필요합니다."})
                        continue
                    if len(data) > app_settings.max_ws_frame_bytes:
                        await send_json({"error": "프레임 크기가 허용 한도를 초과했습니다."})
                        continue
                    frame_id += 1
                    if frame_queue.full():
                        with suppress(asyncio.QueueEmpty):
                            frame_queue.get_nowait()
                            frame_queue.task_done()
                    frame_queue.put_nowait((frame_id, data))
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                disconnected.set()

        receiver = asyncio.create_task(receive_frames())
        try:
            while not disconnected.is_set() or not frame_queue.empty():
                try:
                    current_id, data = await asyncio.wait_for(frame_queue.get(), timeout=0.25)
                except TimeoutError:
                    continue
                try:
                    prediction = await asyncio.to_thread(
                        websocket.app.state.detector.predict,
                        data,
                        confidence,
                        iou,
                    )
                    fire_count, smoke_count = count_classes(prediction.detections)
                    await send_json(
                        {
                            "frame_id": current_id,
                            "image_width": prediction.width,
                            "image_height": prediction.height,
                            "fire_count": fire_count,
                            "smoke_count": smoke_count,
                            "inference_ms": round(prediction.inference_ms, 2),
                            "detections": prediction.detections,
                        }
                    )
                except InvalidImageError as exc:
                    await send_json({"frame_id": current_id, "error": str(exc)})
                except DetectorUnavailableError as exc:
                    await send_json({"frame_id": current_id, "error": str(exc)})
                    await websocket.close(code=1011)
                    break
                except (WebSocketDisconnect, RuntimeError):
                    break
                finally:
                    frame_queue.task_done()
        finally:
            receiver.cancel()
            with suppress(asyncio.CancelledError):
                await receiver

    @app.exception_handler(Exception)
    async def unhandled_exception(_: Request, exc: Exception):
        logger.exception("처리되지 않은 서버 오류", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "서버 처리 중 오류가 발생했습니다."})

    # API와 WebSocket 라우트 등록 뒤 SPA를 마지막에 마운트한다.
    if app_settings.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=app_settings.frontend_dist, html=True), name="frontend")
    return app


app = create_app()
