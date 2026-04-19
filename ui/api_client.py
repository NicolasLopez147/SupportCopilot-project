from __future__ import annotations

from typing import Any

import httpx

from ui.config import UISettings


class GatewayClient:
    def __init__(self, settings: UISettings) -> None:
        self._settings = settings

    def get_health(self) -> dict[str, Any]:
        with httpx.Client(timeout=self._settings.request_timeout_seconds) as client:
            response = client.get(f"{self._settings.gateway_base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()

    def run_copilot(self, payload: dict[str, Any], request_id: str) -> tuple[dict[str, Any], str | None]:
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        }
        with httpx.Client(timeout=self._settings.request_timeout_seconds) as client:
            response = client.post(
                f"{self._settings.gateway_base_url}/api/v1/copilot/run",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json(), response.headers.get("X-Request-ID")
