from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from services.gateway_service.app.main import app as gateway_app
from services.intent_service.app.main import app as intent_app
from services.reply_service.app.main import app as reply_app
from services.summary_service.app.main import app as summary_app
from shared.utils.request_id import REQUEST_ID_HEADER
import ui.app  # noqa: F401


def assert_health_endpoint(app, expected_service: str) -> None:
    client = TestClient(app)
    request_id = f"smoke-{expected_service}"

    response = client.get("/api/v1/health", headers={REQUEST_ID_HEADER: request_id})
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "ok", payload
    assert payload["service"] == expected_service, payload
    assert response.headers.get(REQUEST_ID_HEADER) == request_id, response.headers


def assert_root_endpoint(app, expected_service: str) -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["service"] == expected_service, payload
    assert "docs" in payload, payload
    assert "health" in payload, payload


def main() -> None:
    service_apps = [
        (gateway_app, "gateway-service"),
        (intent_app, "intent-service"),
        (summary_app, "summary-service"),
        (reply_app, "reply-service"),
    ]

    for app, expected_service in service_apps:
        assert_root_endpoint(app, expected_service)
        assert_health_endpoint(app, expected_service)

    print("smoke-tests-ok")


if __name__ == "__main__":
    main()
