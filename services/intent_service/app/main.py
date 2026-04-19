from fastapi import FastAPI, HTTPException, Request, Response

from services.intent_service.app.service import get_intent_service
from shared.config.settings import get_intent_service_settings
from shared.logging.json_logger import build_json_logger
from shared.schemas.gateway import ErrorResponse, HealthResponse
from shared.schemas.intent import IntentRequest, IntentResponse
from shared.utils.request_id import REQUEST_ID_HEADER, resolve_request_id


settings = get_intent_service_settings()
logger = build_json_logger(settings.service_name, settings.log_level)

app = FastAPI(
    title="SupportCopilot Intent Service",
    description="Intent classification microservice for the SupportCopilot architecture.",
    version="0.1.0",
)


def request_to_sample(payload: IntentRequest) -> dict:
    return {
        "conversation_id": payload.conversation_id,
        "scenario": payload.scenario,
        "messages": [message.model_dump() for message in payload.messages],
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
        "intent": "/api/v1/intent",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        mode="service",
    )


@app.post("/api/v1/intent", response_model=IntentResponse, tags=["intent"])
def predict_intent_endpoint(payload: IntentRequest, request: Request, response: Response) -> IntentResponse:
    request_id = request.state.request_id
    response.headers[REQUEST_ID_HEADER] = request_id
    persist_feedback = payload.persist_feedback or settings.enable_feedback_persistence

    try:
        service = get_intent_service()
        sample = request_to_sample(payload)
        result = service.run(sample=sample, persist_feedback=persist_feedback)
        return IntentResponse(**result)
    except ValueError as exc:
        raise make_error(400, "BAD_REQUEST", str(exc), request_id) from exc
    except FileNotFoundError as exc:
        raise make_error(503, "SERVICE_UNAVAILABLE", str(exc), request_id) from exc
    except RuntimeError as exc:
        raise make_error(503, "SERVICE_UNAVAILABLE", str(exc), request_id) from exc
    except Exception as exc:
        logger.exception(
            "intent service failed",
            extra={
                "service": settings.service_name,
                "request_id": request_id,
                "route": "/api/v1/intent",
                "event": "request_failed",
            },
        )
        raise make_error(500, "INTERNAL_ERROR", str(exc), request_id) from exc
