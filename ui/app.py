from __future__ import annotations

from uuid import uuid4

import httpx
import streamlit as st

from ui.api_client import GatewayClient
from ui.components import render_health_panel, render_result_panel, render_technical_panel
from ui.config import get_settings


def _ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "speaker": "customer",
                "text": "My internet keeps freezing and I do not know if this belongs to billing or technical support.",
            },
            {
                "speaker": "agent",
                "text": "I can help you figure that out. Can you tell me more about the issue?",
            },
        ]


def _build_payload(conversation_id: str, scenario: str, persist_feedback: bool) -> dict:
    cleaned_messages = []
    for message in st.session_state.messages:
        text = (message.get("text") or "").strip()
        speaker = (message.get("speaker") or "").strip().lower()
        if text and speaker:
            cleaned_messages.append({"speaker": speaker, "text": text})

    return {
        "conversation_id": conversation_id.strip(),
        "scenario": scenario.strip() or None,
        "persist_feedback": persist_feedback,
        "messages": cleaned_messages,
    }


def _render_message_editor() -> None:
    st.markdown("### Messages")
    for index, message in enumerate(st.session_state.messages):
        with st.container(border=True):
            col1, col2 = st.columns([1, 6])
            with col1:
                speaker = st.selectbox(
                    "Role",
                    options=["customer", "agent"],
                    index=0 if message.get("speaker") == "customer" else 1,
                    key=f"speaker_{index}",
                    label_visibility="collapsed",
                )
            with col2:
                text = st.text_area(
                    f"Message {index + 1}",
                    value=message.get("text", ""),
                    key=f"text_{index}",
                    height=120,
                )
            message["speaker"] = speaker
            message["text"] = text

    col_add, col_remove = st.columns(2)
    with col_add:
        if st.button("Ajouter un message", use_container_width=True):
            st.session_state.messages.append({"speaker": "customer", "text": ""})
            st.rerun()
    with col_remove:
        can_remove = len(st.session_state.messages) > 1
        if st.button("Supprimer le dernier", disabled=not can_remove, use_container_width=True):
            st.session_state.messages.pop()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="SupportCopilot UI", page_icon=":toolbox:", layout="wide")
    settings = get_settings()
    client = GatewayClient(settings)
    _ensure_session_state()

    st.title("SupportCopilot UI")
    st.caption("Interface Streamlit connectee uniquement au gateway-service.")

    health_payload = None
    health_error = None
    with st.sidebar:
        st.header("Configuration")
        st.write(f"Gateway: `{settings.gateway_base_url}`")
        st.write(f"Timeout: `{settings.request_timeout_seconds}` secondes")

        if st.button("Verifier le gateway", use_container_width=True):
            try:
                health_payload = client.get_health()
            except httpx.HTTPError as exc:
                health_error = str(exc)

        render_health_panel(health_payload, health_error)

    left_col, right_col = st.columns([1.1, 1.4], gap="large")

    with left_col:
        st.subheader("Entree")
        conversation_id = f"ui_{uuid4().hex[:12]}"
        persist_feedback = st.checkbox("Persister le feedback", value=False)

        with st.expander("Options avancees", expanded=False):
            st.caption("Ces champs sont optionnels et plutot destines au debug ou a l'experimentation.")
            scenario = st.text_input("Scenario override", value="")
            st.text_input("Conversation ID", value=conversation_id, disabled=True)

        _render_message_editor()

        payload = _build_payload(conversation_id, scenario, persist_feedback)
        has_messages = bool(payload["messages"])

        if st.button("Lancer SupportCopilot", type="primary", disabled=not has_messages, use_container_width=True):
            request_id = f"ui-{uuid4().hex}"
            st.session_state.last_error = None
            try:
                with st.spinner("Execution du pipeline via le gateway-service..."):
                    result, response_request_id = client.run_copilot(payload, request_id)
                st.session_state.last_result = result
                st.session_state.last_request_id = response_request_id or request_id
            except httpx.HTTPStatusError as exc:
                response_text = exc.response.text if exc.response is not None else str(exc)
                st.session_state.last_error = response_text
                st.session_state.last_result = None
            except httpx.HTTPError as exc:
                st.session_state.last_error = str(exc)
                st.session_state.last_result = None

    with right_col:
        st.subheader("Sortie")
        result = st.session_state.get("last_result")
        error_message = st.session_state.get("last_error")

        if error_message and not result:
            st.error(error_message)

        if result:
            render_result_panel(result)
            render_technical_panel(result)
        elif not error_message:
            st.info("Lance une conversation depuis le panneau de gauche pour voir le resultat.")


if __name__ == "__main__":
    main()
