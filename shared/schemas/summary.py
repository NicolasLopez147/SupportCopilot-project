from pydantic import BaseModel, Field

from shared.schemas.gateway import ConversationMessage


class SummaryRequest(BaseModel):
    conversation_id: str | None = Field(default=None, description="Optional conversation identifier.")
    scenario: str | None = Field(default=None, description="Optional scenario label.")
    messages: list[ConversationMessage] = Field(..., min_length=1, description="Conversation messages in chronological order.")
    persist_feedback: bool = Field(default=False, description="Enable critic failure persistence for this request.")


class SummaryReview(BaseModel):
    passed: bool
    score: float | None = None
    issues: list[str] = Field(default_factory=list)
    fallback_summary: str
    final_summary: str
    used_fallback: bool = False


class SummaryResponse(BaseModel):
    conversation_id: str | None = None
    scenario: str | None = None
    conversation_text: str
    summary_raw: str
    summary_review: SummaryReview
    summary: str
