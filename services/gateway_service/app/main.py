from fastapi import FastAPI, HTTPException, Request, Response

from services.gateway_service.app.orchestrator import (
    GatewayOrchestrationError,
    get_embedded_gateway_orchestrator,
    get_http_gateway_orchestrator,
)
from shared.config.settings import get_gateway_settings
from shared.logging.json_logger import build_json_logger
from shared.schemas.gateway import CopilotBatchRequest, CopilotRunRequest, ErrorResponse, HealthResponse
from shared.utils.request_id import REQUEST_ID_HEADER, resolve_request_id


settings = get_gateway_settings()
logger = build_json_logger(settings.service_name, settings.log_level)

app = FastAPI(
    title="SupportCopilot Gateway API",
    description="Gateway service for the SupportCopilot microservices architecture.",
    version="0.1.0",
)


def request_to_sample(payload: CopilotRunRequest) -> dict:
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
        "mode": "embedded" if settings.gateway_use_embedded_mode else "service",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        mode="embedded" if settings.gateway_use_embedded_mode else "service",
    )


def get_gateway_orchestrator():
    if settings.gateway_use_embedded_mode:
        return get_embedded_gateway_orchestrator()
    return get_http_gateway_orchestrator(
        intent_service_url=settings.intent_service_url,
        summary_service_url=settings.summary_service_url,
        reply_service_url=settings.reply_service_url,
        intent_timeout_seconds=settings.intent_timeout_seconds,
        summary_timeout_seconds=settings.summary_timeout_seconds,
        reply_timeout_seconds=settings.reply_timeout_seconds,
    )


@app.post("/api/v1/copilot/run", tags=["gateway"])
def run_copilot(payload: CopilotRunRequest, request: Request, response: Response) -> dict:
    request_id = request.state.request_id
    response.headers[REQUEST_ID_HEADER] = request_id

    try:
        orchestrator = get_gateway_orchestrator()
        sample = request_to_sample(payload)
        result = orchestrator.run_one(
            sample=sample,
            persist_feedback=payload.persist_feedback,
            request_id=request_id,
        )
        result["request_id"] = request_id
        return result
    except ValueError as exc:
        raise make_error(400, "BAD_REQUEST", str(exc), request_id) from exc
    except FileNotFoundError as exc:
        raise make_error(503, "SERVICE_UNAVAILABLE", str(exc), request_id) from exc
    except GatewayOrchestrationError as exc:
        raise make_error(exc.status_code, exc.code, exc.message, request_id) from exc
    except Exception as exc:
        logger.exception(
            "gateway orchestration failed",
            extra={
                "service": settings.service_name,
                "request_id": request_id,
                "route": "/api/v1/copilot/run",
                "event": "request_failed",
            },
        )
        raise make_error(500, "INTERNAL_ERROR", str(exc), request_id) from exc


@app.post("/api/v1/copilot/run/batch", tags=["gateway"])
def run_copilot_batch(payload: CopilotBatchRequest, request: Request, response: Response) -> dict:
    request_id = request.state.request_id
    response.headers[REQUEST_ID_HEADER] = request_id

    try:
        orchestrator = get_gateway_orchestrator()
        samples = [request_to_sample(item) for item in payload.conversations]
        persist_feedback = any(item.persist_feedback for item in payload.conversations)
        results = orchestrator.run_batch(
            samples=samples,
            persist_feedback=persist_feedback,
            request_id=request_id,
        )
        return {
            "count": len(results),
            "request_id": request_id,
            "results": results,
        }
    except ValueError as exc:
        raise make_error(400, "BAD_REQUEST", str(exc), request_id) from exc
    except FileNotFoundError as exc:
        raise make_error(503, "SERVICE_UNAVAILABLE", str(exc), request_id) from exc
    except GatewayOrchestrationError as exc:
        raise make_error(exc.status_code, exc.code, exc.message, request_id) from exc
    except Exception as exc:
        logger.exception(
            "gateway batch orchestration failed",
            extra={
                "service": settings.service_name,
                "request_id": request_id,
                "route": "/api/v1/copilot/run/batch",
                "event": "request_failed",
            },
        )
        raise make_error(500, "INTERNAL_ERROR", str(exc), request_id) from exc
