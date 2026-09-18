from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app


def test_health_and_model_info(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": True, "device": "cpu"}
    assert client.get("/api/model-info").json()["classes"] == {"0": "fire", "1": "smoke"}


def test_valid_image_upload_and_download(client, image_bytes):
    response = client.post("/api/detect/image", files={"file": ("sample.jpg", image_bytes, "image/jpeg")}, data={"confidence": "0.4", "iou": "0.5"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["fire_count"] == 1 and payload["smoke_count"] == 0
    assert (payload["image_width"], payload["image_height"]) == (32, 24)
    result = client.get(payload["result_image_url"])
    assert result.status_code == 200 and result.headers["content-type"] == "image/jpeg"


def test_empty_and_non_image_upload(client):
    empty = client.post("/api/detect/image", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400
    text = client.post("/api/detect/image", files={"file": ("x.txt", b"hello", "text/plain")})
    assert text.status_code == 415
    spoofed = client.post("/api/detect/image", files={"file": ("x.jpg", b"hello", "image/jpeg")})
    assert spoofed.status_code == 400


def test_corrupt_image_and_size_limit(client):
    corrupt = client.post("/api/detect/image", files={"file": ("x.jpg", b"not-image", "image/jpeg")})
    assert corrupt.status_code == 400
    too_large = client.post("/api/detect/image", files={"file": ("x.jpg", b"0" * (1024 * 1024 + 1), "image/jpeg")})
    assert too_large.status_code == 413


def test_threshold_validation(client, image_bytes):
    files = {"file": ("sample.jpg", image_bytes, "image/jpeg")}
    assert client.post("/api/detect/image", files=files, data={"confidence": "1.1"}).status_code == 422
    files = {"file": ("sample.jpg", image_bytes, "image/jpeg")}
    assert client.post("/api/detect/image", files=files, data={"iou": "-0.1"}).status_code == 422


def test_websocket_valid_invalid_and_close(client, image_bytes):
    with client.websocket_connect("/ws/detect") as websocket:
        websocket.send_text("not binary")
        assert "error" in websocket.receive_json()
        websocket.send_bytes(image_bytes)
        payload = websocket.receive_json()
        assert payload["frame_id"] == 1
        assert payload["fire_count"] == 1


def test_websocket_bad_threshold(client):
    with client.websocket_connect("/ws/detect?confidence=9") as websocket:
        assert "error" in websocket.receive_json()


def test_model_missing_returns_degraded_health_and_503(tmp_path, image_bytes):
    config = Settings(
        model_path=tmp_path / "missing.pt",
        results_dir=tmp_path / "results",
        frontend_dist=tmp_path / "dist",
    )
    with TestClient(create_app(config)) as missing_client:
        health = missing_client.get("/api/health").json()
        assert health["status"] == "degraded"
        assert health["model_ready"] is False
        info = missing_client.get("/api/model-info").json()
        assert "모델 파일이 없습니다" in info["error"]
        response = missing_client.post(
            "/api/detect/image",
            files={"file": ("sample.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
