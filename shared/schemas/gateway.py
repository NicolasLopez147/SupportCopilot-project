from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    speaker: str = Field(..., description="Message speaker, usually 'customer' or 'agent'.")
    text: str = Field(..., description="Raw message text.")


class CopilotRunRequest(BaseModel):
    conversation_id: str | None = Field(default=None, description="Optional conversation identifier.")
    scenario: str | None = Field(default=None, description="Optional scenario label.")
    messages: list[ConversationMessage] = Field(..., min_length=1, description="Conversation messages in chronological order.")
    persist_feedback: bool = Field(default=False, description="Enable critic failure persistence for this request.")


class CopilotBatchRequest(BaseModel):
    conversations: list[CopilotRunRequest] = Field(..., min_length=1, description="Batch of conversations to process.")


class HealthResponse(BaseModel):
    status: str
    service: str
    mode: str


class ErrorPayload(BaseModel):
    code: str
    message: str
    service: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorPayload
