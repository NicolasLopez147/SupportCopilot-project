from pydantic import BaseModel, Field

from shared.schemas.gateway import ConversationMessage


class ReplyRequest(BaseModel):
    conversation_id: str | None = Field(default=None, description="Optional conversation identifier.")
    scenario: str | None = Field(default=None, description="Optional scenario label.")
    messages: list[ConversationMessage] = Field(..., min_length=1, description="Conversation messages in chronological order.")
    predicted_intent: str = Field(..., description="Predicted intent produced upstream by the intent service.")
    summary_text: str = Field(..., min_length=1, description="Summary produced upstream by the summary service.")
    persist_feedback: bool = Field(default=False, description="Enable critic failure persistence for this request.")


class ReplyReview(BaseModel):
    passed: bool
    score: float | None = None
    issues: list[str] = Field(default_factory=list)
    fallback_reply: str | None = None
    final_reply: str
    used_fallback: bool = False


class ReplyResponse(BaseModel):
    conversation_id: str | None = None
    scenario: str | None = None
    conversation_text: str
    suggested_reply_raw: str
    reply_review: ReplyReview
    suggested_reply: str
