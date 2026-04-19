from __future__ import annotations

from typing import Any

import streamlit as st


def render_health_panel(health_payload: dict[str, Any] | None, error_message: str | None) -> None:
    st.subheader("Etat du gateway")
    if error_message:
        st.error(error_message)
        return

    if not health_payload:
        st.info("Aucune verification n'a encore ete effectuee.")
        return

    status = health_payload.get("status", "unknown")
    service = health_payload.get("service", "gateway-service")
    if status == "ok":
        st.success(f"{service}: {status}")
    else:
        st.warning(f"{service}: {status}")


def render_review_block(title: str, review_payload: dict[str, Any] | None) -> None:
    with st.expander(title, expanded=False):
        if not review_payload:
            st.info("Aucune revue disponible.")
            return

        passed = review_payload.get("passed")
        score = review_payload.get("score")
        issues = review_payload.get("issues", [])
        used_fallback = review_payload.get("used_fallback", False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Passe", "Oui" if passed else "Non")
        col2.metric("Score", f"{score:.2f}" if isinstance(score, (int, float)) else "-")
        col3.metric("Fallback", "Oui" if used_fallback else "Non")

        if issues:
            st.markdown("**Problemes detectes**")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.write("Aucun probleme detecte.")

        st.json(review_payload)


def render_result_panel(result: dict[str, Any]) -> None:
    st.subheader("Resultat principal")
    col1, col2, col3 = st.columns(3)

    predicted_intent = ((result.get("intent") or {}).get("predicted_intent")) or "N/A"
    summary_text = result.get("summary") or "N/A"
    reply_text = result.get("suggested_reply") or "N/A"

    with col1:
        st.markdown("**Intent**")
        st.code(predicted_intent, language=None)

    with col2:
        st.markdown("**Request ID**")
        st.code(result.get("request_id", "N/A"), language=None)

    with col3:
        st.metric(
            "Fallback reply",
            "Oui" if ((result.get("reply_review") or {}).get("used_fallback")) else "Non",
        )

    st.markdown("**Resume final**")
    st.write(summary_text)

    st.markdown("**Reponse finale suggeree**")
    st.write(reply_text)


def render_technical_panel(result: dict[str, Any]) -> None:
    st.subheader("Details techniques")

    with st.expander("Intent brut", expanded=False):
        st.json(result.get("intent_raw", {}))
    render_review_block("Intent critic", result.get("intent_review"))

    with st.expander("Summary brut", expanded=False):
        st.write(result.get("summary_raw", ""))
    render_review_block("Summary critic", result.get("summary_review"))

    with st.expander("Reply brute", expanded=False):
        st.write(result.get("suggested_reply_raw", ""))
    render_review_block("Reply critic", result.get("reply_review"))
