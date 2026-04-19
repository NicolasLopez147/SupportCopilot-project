from dataclasses import dataclass
import os


@dataclass(frozen=True)
class UISettings:
    gateway_base_url: str = os.getenv("GATEWAY_SERVICE_URL", "http://127.0.0.1:8000")
    request_timeout_seconds: float = float(os.getenv("UI_REQUEST_TIMEOUT_SECONDS", "120"))


def get_settings() -> UISettings:
    return UISettings()
