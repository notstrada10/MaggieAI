"""Streamlit trace-inspection workbench for MaggieAI.

Single page:
- Latin sentence input + Translate button
- Translation rendered prominently
- Tabs: Rationale | Morphology | Evidence | Raw Trace
- Sidebar: 20 most recent traces, click any to load it back

External dependencies:
- Gateway HTTP API (``GATEWAY_URL``, default ``http://localhost:18000``)
  for POST /translate
- Postgres (via :mod:`maggieai.db.engine`) for trace history lookup

Streamlit re-runs this entire script on every interaction, so:
- Connection-like resources (httpx client, DB engine) live behind
  ``@st.cache_resource`` so they survive across reruns;
- Per-session values (current input text, last result, loaded trace)
  live in ``st.session_state`` so they don't reset on each click.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

import httpx
import streamlit as st
from sqlalchemy import desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from maggieai.db.engine import get_engine
from maggieai.db.models import ReasoningTrace


# -------------------------------------------------------------------
# Cached resources (one per process, shared across reruns)
# -------------------------------------------------------------------
@st.cache_resource
def _http_client() -> httpx.Client:
    return httpx.Client(timeout=300.0)


@st.cache_resource
def _db_engine() -> Engine:
    return get_engine()


def _gateway_url() -> str:
    return os.environ.get("GATEWAY_URL", "http://localhost:18000")


# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------
def call_translate(text: str) -> dict[str, Any]:
    resp = _http_client().post(f"{_gateway_url()}/translate", json={"text": text})
    resp.raise_for_status()
    body: dict[str, Any] = resp.json()
    return body


def fetch_recent_traces(limit: int = 20) -> list[dict[str, Any]]:
    with Session(_db_engine()) as session:
        stmt = select(ReasoningTrace).order_by(desc(ReasoningTrace.created_at)).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "trace_id": str(r.trace_id),
                "input_text": r.input_text,
                "created_at": r.created_at.isoformat(timespec="seconds"),
                "state_dump": r.state_dump,
            }
            for r in rows
        ]


def fetch_trace_by_id(trace_id: str) -> dict[str, Any] | None:
    with Session(_db_engine()) as session:
        row = session.get(ReasoningTrace, UUID(trace_id))
        if row is None:
            return None
        return {
            "trace_id": str(row.trace_id),
            "input_text": row.input_text,
            "created_at": row.created_at.isoformat(timespec="seconds"),
            "state_dump": row.state_dump,
        }


# -------------------------------------------------------------------
# Renderers
# -------------------------------------------------------------------
def render_translation(result: dict[str, Any]) -> None:
    translation = (result.get("translation") or "").strip()
    if translation:
        st.markdown(f"### {translation}")
    else:
        # The translator returned an empty string. By contract (see
        # prompts/system_translator.j2 rule 6) this means the input was
        # not recognized as Latin and the model refused. The rationale
        # tab carries the explanation.
        st.warning(
            "**Could not translate.** The model declined this input "
            "(likely not Latin). Open the **Rationale** tab to see why."
        )
    st.caption(
        f"trace_id: `{result.get('trace_id', '?')}` · iterations: "
        f"{result.get('iterations', '?')}"
    )


def render_rationale(result: dict[str, Any]) -> None:
    rationale = result.get("rationale") or "_(no rationale)_"
    st.markdown(rationale)


# Keys we surface in the human-readable morphology table. UD canonical
# features only — CLTK-internal taxonomy (InflClass, NameType,
# PronominalType, the stray "<class 'NoneType'>" key) is filtered out
# here. The full feature dict is still available under the "Raw Trace"
# tab for debugging.
_DISPLAY_FEATURE_KEYS: tuple[str, ...] = (
    "Case",
    "VerbForm",
    "Mood",
    "Tense",
    "Aspect",
    "Voice",
    "Number",
    "Gender",
    "Person",
    "Degree",
)


def _filter_features(raw: dict[str, str] | None) -> str:
    if not raw:
        return ""
    return ", ".join(f"{k}={raw[k]}" for k in _DISPLAY_FEATURE_KEYS if k in raw)


def render_morphology(tokens: list[dict[str, Any]]) -> None:
    if not tokens:
        st.caption("_(no morphology data)_")
        return
    rows = [
        {
            "#": t.get("index", ""),
            "Token": t.get("text", ""),
            "Lemma": t.get("lemma") or "?",
            "POS": t.get("pos") or "?",
            "Features": _filter_features(t.get("features")),
        }
        for t in tokens
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(
        "Features filtered to the UD canonical set. CLTK-internal fields "
        "like `InflClass`, `NameType`, `PronominalType` are visible in the "
        "**Raw Trace** tab."
    )


def render_evidence(result: dict[str, Any]) -> None:
    phenomena = result.get("phenomena_detected", [])
    st.markdown("**Phenomena detected**")
    if phenomena:
        # Render each as a code chip on a single wrapped line
        st.markdown(" ".join(f"`{p}`" for p in phenomena))
    else:
        st.caption("_(none)_")

    st.markdown("---")
    citations = result.get("citations", [])
    st.markdown("**Citations**")
    if not citations:
        st.caption("_(no citations — TM is empty and no grammar matched)_")
        return
    for c in citations:
        if c.get("type") == "grammar_rule":
            st.markdown(f"- 📚 **{c.get('rule', '?')}** — {c.get('source', '?')}")
        elif c.get("type") == "translation_memory":
            distance = c.get("distance", 0.0)
            st.markdown(
                f"- 📖 {c.get('source', '?')} "
                f"(translator: {c.get('translator') or 'anonymous'}, "
                f"distance {distance:.3f})"
            )
        else:
            st.markdown(f"- {c}")


def render_raw_trace(payload: dict[str, Any]) -> None:
    st.json(payload, expanded=False)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="MaggieAI Workbench", layout="wide")
    st.title("MaggieAI — Latin Translation Workbench")
    st.caption(
        f"Gateway: `{_gateway_url()}` · "
        "DB traces from the `reasoning_traces` table"
    )

    # ----- Sidebar: trace history -----
    with st.sidebar:
        st.header("Recent traces")
        try:
            traces = fetch_recent_traces(limit=20)
        except Exception as exc:  # pragma: no cover — surface to UI
            st.error(f"DB unreachable: {exc}")
            traces = []
        if not traces:
            st.caption("_(no traces yet — run a translation)_")
        else:
            for t in traces:
                preview = t["input_text"][:48] + (
                    "..." if len(t["input_text"]) > 48 else ""
                )
                if st.button(
                    f"`{t['created_at']}`\n\n{preview}",
                    key=f"trace_btn_{t['trace_id']}",
                    use_container_width=True,
                ):
                    st.session_state["loaded_trace"] = t
                    st.session_state.pop("last_result", None)

    # ----- Main: input -----
    text = st.text_area(
        "Latin sentence",
        value=st.session_state.get("input_text", ""),
        height=100,
        placeholder="Caesare imperante, Galli rebellaverunt",
    )

    btn_cols = st.columns([1, 1, 6])
    if btn_cols[0].button("Translate", type="primary"):
        if not text.strip():
            st.warning("Enter a sentence first.")
        else:
            st.session_state["input_text"] = text
            with st.spinner("Calling /translate (Claude + morphology + retrieve)..."):
                try:
                    fresh_result = call_translate(text)
                    st.session_state["last_result"] = fresh_result
                    st.session_state.pop("loaded_trace", None)
                except httpx.HTTPError as exc:
                    st.error(f"Gateway error: {exc}")
                    return
    if btn_cols[1].button("Clear"):
        for k in ("last_result", "loaded_trace", "input_text"):
            st.session_state.pop(k, None)
        st.rerun()

    # ----- Main: render result, either fresh or loaded -----
    result: dict[str, Any] | None = None
    state_dump: dict[str, Any] | None = None
    if "loaded_trace" in st.session_state:
        trace = st.session_state["loaded_trace"]
        raw_dump = trace.get("state_dump")
        if isinstance(raw_dump, dict):
            state_dump = raw_dump
            raw_output = state_dump.get("output")
            if isinstance(raw_output, dict):
                result = raw_output
        st.info(
            f"Viewing historical trace from `{trace['created_at']}` — "
            f"input: _{trace['input_text']}_"
        )
    elif "last_result" in st.session_state:
        last = st.session_state["last_result"]
        if isinstance(last, dict):
            result = last

    if not result:
        st.caption("_(translate something or load a trace from the sidebar)_")
        return

    st.markdown("---")
    render_translation(result)

    rationale_tab, morph_tab, evidence_tab, trace_tab = st.tabs(
        ["Rationale", "Morphology", "Evidence", "Raw Trace"]
    )
    with rationale_tab:
        render_rationale(result)
    with morph_tab:
        render_morphology(result.get("morpho_analysis", []))
    with evidence_tab:
        render_evidence(result)
    with trace_tab:
        # Prefer the full state_dump when we have it (loaded from history),
        # otherwise show what we have from the live response.
        render_raw_trace(state_dump if state_dump is not None else result)


main()
