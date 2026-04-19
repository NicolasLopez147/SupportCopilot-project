from pydantic import BaseModel, Field

from shared.schemas.gateway import ConversationMessage


class IntentRequest(BaseModel):
    conversation_id: str | None = Field(default=None, description="Optional conversation identifier.")
    scenario: str | None = Field(default=None, description="Optional scenario label.")
    messages: list[ConversationMessage] = Field(..., min_length=1, description="Conversation messages in chronological order.")
    persist_feedback: bool = Field(default=False, description="Enable critic failure persistence for this request.")


class IntentTopClass(BaseModel):
    label: str
    score: float


class IntentPrediction(BaseModel):
    input_text: str
    predicted_intent: str
    confidence: float | None = None
    top_classes: list[IntentTopClass] = Field(default_factory=list)


class IntentReview(BaseModel):
    passed: bool
    score: float | None = None
    issues: list[str] = Field(default_factory=list)
    suggested_intent: str | None = None
    final_intent: str
    used_fallback: bool = False


class IntentResponse(BaseModel):
    conversation_id: str | None = None
    scenario: str | None = None
    conversation_text: str
    intent_raw: IntentPrediction
    intent_review: IntentReview
    intent: IntentPrediction
