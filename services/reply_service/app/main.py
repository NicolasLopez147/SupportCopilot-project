from fastapi import FastAPI, HTTPException, Request, Response

from services.reply_service.app.service import get_reply_service
from shared.config.settings import get_reply_service_settings
from shared.logging.json_logger import build_json_logger
from shared.schemas.gateway import ErrorResponse, HealthResponse
from shared.schemas.reply import ReplyRequest, ReplyResponse
from shared.utils.request_id import REQUEST_ID_HEADER, resolve_request_id


settings = get_reply_service_settings()
logger = build_json_logger(settings.service_name, settings.log_level)

app = FastAPI(
    title="SupportCopilot Reply Service",
    description="Reply generation microservice for the SupportCopilot architecture.",
    version="0.1.0",
)


def request_to_sample(payload: ReplyRequest) -> dict:
    return {
        "conversation_id": payload.conversation_id,
        "scenario": payload.scenario,
        "messages": [message.model_dump() for message in payload.messages],
        "predicted_intent": payload.predicted_intent,
        "summary_text": payload.summary_text,
    }


def make_error(status_code: int, code: str, message: str, request_id: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(
            error={
                "code": code,
                "message": message,
                "service": settings.service_name,
                "request_id": request_id,
            }
        ).model_dump(),
    )


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id

    logger.info(
        "incoming request",
        extra={
            "service": settings.service_name,
            "request_id": request_id,
            "route": request.url.path,
            "event": "request_started",
        },
    )

    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": settings.service_name,
        "docs": "/docs",
        "health": "/api/v1/health",
        "reply": "/api/v1/reply",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        mode="service",
    )


@app.post("/api/v1/reply", response_model=ReplyResponse, tags=["reply"])
def generate_reply_endpoint(payload: ReplyRequest, request: Request, response: Response) -> ReplyResponse:
    request_id = request.state.request_id
    response.headers[REQUEST_ID_HEADER] = request_id
    persist_feedback = payload.persist_feedback or settings.enable_feedback_persistence

    try:
        service = get_reply_service()
        sample = request_to_sample(payload)
        result = service.run(sample=sample, persist_feedback=persist_feedback)
        return ReplyResponse(**result)
    except ValueError as exc:
        raise make_error(400, "BAD_REQUEST", str(exc), request_id) from exc
    except FileNotFoundError as exc:
        raise make_error(503, "SERVICE_UNAVAILABLE", str(exc), request_id) from exc
    except RuntimeError as exc:
        raise make_error(503, "SERVICE_UNAVAILABLE", str(exc), request_id) from exc
    except Exception as exc:
        logger.exception(
            "reply service failed",
            extra={
                "service": settings.service_name,
                "request_id": request_id,
                "route": "/api/v1/reply",
                "event": "request_failed",
            },
        )
        raise make_error(500, "INTERNAL_ERROR", str(exc), request_id) from exc
