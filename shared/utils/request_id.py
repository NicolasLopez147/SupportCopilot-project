from uuid import uuid4


REQUEST_ID_HEADER = "X-Request-ID"


def generate_request_id() -> str:
    return f"req-{uuid4().hex}"


def resolve_request_id(raw_request_id: str | None) -> str:
    cleaned = (raw_request_id or "").strip()
    return cleaned or generate_request_id()
