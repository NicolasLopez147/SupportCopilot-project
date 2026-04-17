from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    speaker: str
    text: str


class StructuredSummary(BaseModel):
    issue: Optional[str] = None
    context: Optional[str] = None
    actions_already_tried: List[str] = Field(default_factory=list)
    next_best_action: Optional[str] = None


class SupportSample(BaseModel):
    
    conversation_id: str
    source: str
    language: str = "en"
    channel: str
    messages: List[Message]

    intent_label: Optional[str] = None
    summary_structured: Optional[StructuredSummary] = None
    summary_abstractive: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)