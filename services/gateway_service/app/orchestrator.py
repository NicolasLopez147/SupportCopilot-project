from functools import lru_cache

import httpx

from src.copilot.pipeline.service import SupportCopilotService


class GatewayOrchestrationError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class EmbeddedGatewayOrchestrator:
    def __init__(self) -> None:
        self._service = SupportCopilotService()

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "gateway-service",
            "mode": "embedded",
        }

    def run_one(self, sample: dict, persist_feedback: bool, request_id: str) -> dict:
        return self._service.run_sample(sample=sample, persist_feedback=persist_feedback)

    def run_batch(self, samples: list[dict], persist_feedback: bool, request_id: str) -> list[dict]:
        return self._service.run_samples(samples=samples, persist_feedback=persist_feedback)


class HttpGatewayOrchestrator:
    def __init__(
        self,
        intent_service_url: str,
        summary_service_url: str,
        reply_service_url: str,
        intent_timeout_seconds: float,
        summary_timeout_seconds: float,
        reply_timeout_seconds: float,
    ) -> None:
        self.intent_service_url = intent_service_url.rstrip("/")
        self.summary_service_url = summary_service_url.rstrip("/")
        self.reply_service_url = reply_service_url.rstrip("/")
        self.intent_timeout_seconds = intent_timeout_seconds
        self.summary_timeout_seconds = summary_timeout_seconds
        self.reply_timeout_seconds = reply_timeout_seconds

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "gateway-service",
            "mode": "service",
        }

    def _post_json(self, url: str, payload: dict, timeout_seconds: float, request_id: str) -> dict:
        try:
            response = httpx.post(
                url,
                json=payload,
                headers={"X-Request-ID": request_id},
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise GatewayOrchestrationError(
                status_code=504,
                code="DOWNSTREAM_TIMEOUT",
                message=f"Downstream timeout while calling {url}.",
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewayOrchestrationError(
                status_code=503,
                code="DOWNSTREAM_UNAVAILABLE",
                message=f"Could not reach downstream service at {url}.",
            ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise GatewayOrchestrationError(
                status_code=502,
                code="INVALID_DOWNSTREAM_RESPONSE",
                message=f"Downstream service at {url} returned a non-JSON response.",
            ) from exc

        if response.status_code >= 400:
            detail = body.get("detail") if isinstance(body, dict) else None
            if isinstance(detail, dict) and "error" in detail:
                downstream_error = detail["error"]
                message = downstream_error.get("message", f"Downstream service at {url} failed.")
            else:
                message = str(detail or body)

            status_code = 504 if response.status_code == 504 else 503
            raise GatewayOrchestrationError(
                status_code=status_code,
                code="DOWNSTREAM_SERVICE_ERROR",
                message=message,
            )

        if not isinstance(body, dict):
            raise GatewayOrchestrationError(
                status_code=502,
                code="INVALID_DOWNSTREAM_RESPONSE",
                message=f"Downstream service at {url} returned an invalid payload shape.",
            )

        return body

    def run_one(self, sample: dict, persist_feedback: bool, request_id: str) -> dict:
        base_payload = {
            "conversation_id": sample.get("conversation_id"),
            "scenario": sample.get("scenario"),
            "messages": sample.get("messages", []),
            "persist_feedback": persist_feedback,
        }

        intent_result = self._post_json(
            url=f"{self.intent_service_url}/api/v1/intent",
            payload=base_payload,
            timeout_seconds=self.intent_timeout_seconds,
            request_id=request_id,
        )
        summary_result = self._post_json(
            url=f"{self.summary_service_url}/api/v1/summary",
            payload=base_payload,
            timeout_seconds=self.summary_timeout_seconds,
            request_id=request_id,
        )
        reply_payload = {
            **base_payload,
            "predicted_intent": intent_result["intent"]["predicted_intent"],
            "summary_text": summary_result["summary"],
        }
        reply_result = self._post_json(
            url=f"{self.reply_service_url}/api/v1/reply",
            payload=reply_payload,
            timeout_seconds=self.reply_timeout_seconds,
            request_id=request_id,
        )

        return {
            "conversation_id": sample.get("conversation_id"),
            "scenario": sample.get("scenario"),
            "conversation_text": summary_result["conversation_text"],
            "intent_raw": intent_result["intent_raw"],
            "intent_review": intent_result["intent_review"],
            "intent": intent_result["intent"],
            "summary_raw": summary_result["summary_raw"],
            "summary_review": summary_result["summary_review"],
            "summary": summary_result["summary"],
            "suggested_reply_raw": reply_result["suggested_reply_raw"],
            "reply_review": reply_result["reply_review"],
            "suggested_reply": reply_result["suggested_reply"],
        }

    def run_batch(self, samples: list[dict], persist_feedback: bool, request_id: str) -> list[dict]:
        return [self.run_one(sample=sample, persist_feedback=persist_feedback, request_id=request_id) for sample in samples]


@lru_cache(maxsize=1)
def get_embedded_gateway_orchestrator() -> EmbeddedGatewayOrchestrator:
    return EmbeddedGatewayOrchestrator()


@lru_cache(maxsize=1)
def get_http_gateway_orchestrator(
    intent_service_url: str,
    summary_service_url: str,
    reply_service_url: str,
    intent_timeout_seconds: float,
    summary_timeout_seconds: float,
    reply_timeout_seconds: float,
) -> HttpGatewayOrchestrator:
    return HttpGatewayOrchestrator(
        intent_service_url=intent_service_url,
        summary_service_url=summary_service_url,
        reply_service_url=reply_service_url,
        intent_timeout_seconds=intent_timeout_seconds,
        summary_timeout_seconds=summary_timeout_seconds,
        reply_timeout_seconds=reply_timeout_seconds,
    )
