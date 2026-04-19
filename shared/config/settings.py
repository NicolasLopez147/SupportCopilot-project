import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GatewaySettings:
    service_name: str = "gateway-service"
    host: str = os.getenv("GATEWAY_HOST", "127.0.0.1")
    port: int = int(os.getenv("GATEWAY_PORT", "8000"))
    intent_service_url: str = os.getenv("INTENT_SERVICE_URL", "http://127.0.0.1:8001")
    summary_service_url: str = os.getenv("SUMMARY_SERVICE_URL", "http://127.0.0.1:8002")
    reply_service_url: str = os.getenv("REPLY_SERVICE_URL", "http://127.0.0.1:8003")
    intent_timeout_seconds: float = float(os.getenv("INTENT_SERVICE_TIMEOUT_SECONDS", "10"))
    summary_timeout_seconds: float = float(os.getenv("SUMMARY_SERVICE_TIMEOUT_SECONDS", "45"))
    reply_timeout_seconds: float = float(os.getenv("REPLY_SERVICE_TIMEOUT_SECONDS", "45"))
    enable_feedback_persistence: bool = os.getenv("ENABLE_FEEDBACK_PERSISTENCE", "false").lower() == "true"
    gateway_use_embedded_mode: bool = os.getenv("GATEWAY_USE_EMBEDDED_MODE", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()


@dataclass(frozen=True)
class IntentServiceSettings:
    service_name: str = "intent-service"
    host: str = os.getenv("INTENT_SERVICE_HOST", "127.0.0.1")
    port: int = int(os.getenv("INTENT_SERVICE_PORT", "8001"))
    model_path: str = os.getenv(
        "INTENT_MODEL_PATH",
        "outputs/experiments/intent/synthetic_embedding/intent_synthetic_model.joblib",
    )
    enable_feedback_persistence: bool = os.getenv("ENABLE_FEEDBACK_PERSISTENCE", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_intent_service_settings() -> IntentServiceSettings:
    return IntentServiceSettings()


@dataclass(frozen=True)
class SummaryServiceSettings:
    service_name: str = "summary-service"
    host: str = os.getenv("SUMMARY_SERVICE_HOST", "127.0.0.1")
    port: int = int(os.getenv("SUMMARY_SERVICE_PORT", "8002"))
    model_path: str = os.getenv(
        "SUMMARY_MODEL_PATH",
        "outputs/experiments/summary/lora_base/final_model",
    )
    enable_feedback_persistence: bool = os.getenv("ENABLE_FEEDBACK_PERSISTENCE", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_summary_service_settings() -> SummaryServiceSettings:
    return SummaryServiceSettings()


@dataclass(frozen=True)
class ReplyServiceSettings:
    service_name: str = "reply-service"
    host: str = os.getenv("REPLY_SERVICE_HOST", "127.0.0.1")
    port: int = int(os.getenv("REPLY_SERVICE_PORT", "8003"))
    model_path: str = os.getenv(
        "REPLY_MODEL_PATH",
        "outputs/experiments/reply/lora_base/final_model",
    )
    enable_feedback_persistence: bool = os.getenv("ENABLE_FEEDBACK_PERSISTENCE", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_reply_service_settings() -> ReplyServiceSettings:
    return ReplyServiceSettings()
