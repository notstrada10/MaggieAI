"""Streamlit trace-inspection workbench for MaggieAI.

Layout:
- Sidebar: routing-mode selector, recent traces, compare-mode toggle
- Main: Latin input → tokenized clickable source → translation + rating
        → tabs (Rationale / Morphology / Evidence / Raw Trace)
- Compare mode: two traces side-by-side (input, translation, rationale,
  morphology) for spotting regressions.

Aesthetic: scholarly — parchment background, ink charcoal text, oxblood
accent, system serif stack, restrained scholarly marks (§ for grammar
rules, ¶ for TM citations). Theme tokens live in `.streamlit/config.toml`;
typography and component overrides are in the CSS block below.

Streamlit re-runs this entire script on every interaction, so:
- Connection-like resources (httpx client, DB engine) live behind
  ``@st.cache_resource`` so they survive across reruns;
- Per-session values (current input text, last result, selected token,
  loaded trace, ratings) live in ``st.session_state`` so they don't
  reset on each click.
"""

from __future__ import annotations

import html
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
# Stylesheet — scholarly palette, system serif, sober interactive states
# -------------------------------------------------------------------
_CSS = """
<style>
:root {
  --bg:           #EFE7D3;   /* parchment */
  --surface:     #E6DCC2;   /* parchment-shadow */
  --surface-2:   #DDCFA8;   /* selection (deeper parchment) */
  --ink:         #1B1812;   /* primary text */
  --ink-soft:    #4D4337;   /* body text on light surfaces */
  --muted:       #7A6E5B;   /* captions, distance values */
  --rule:        #2E2519;   /* horizontal rules, table borders */
  --accent:      #7A1F1F;   /* oxblood — primary action, selection */
  --accent-soft: #9B3636;   /* hover */
  --gold:        #A07B2A;   /* rare emphasis (TM relevance) */
  --serif:       'Iowan Old Style', 'Palatino', 'Palatino Linotype',
                 'Book Antiqua', Georgia, serif;
  --mono:        ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}

/* --- Base typography ------------------------------------------- */
html, body, [class*="st-"], .stMarkdown, .stMarkdown p {
  font-family: var(--serif);
  color: var(--ink);
}
.stApp { background: var(--bg); }
h1, h2, h3, h4 { font-family: var(--serif); letter-spacing: 0; font-weight: 600; }
h1 { font-size: 1.85rem; }
h2 { font-size: 1.4rem;  }
h3 { font-size: 1.15rem; }
code, pre, .mono { font-family: var(--mono); font-size: 0.86em; }

/* Streamlit ships Material Symbols as ligature fonts — without an
   explicit font-family on icon spans the parent's serif cascades down
   and the ligature renders as raw text (e.g. "keyboard_double_arrow_left"
   in place of the sidebar-collapse arrow). Pin the icon font here. */
[class*="material-symbols"],
[class*="Material-Symbols"],
.material-icons,
.material-icons-outlined,
[data-testid*="MaterialIcon"],
[data-testid*="stIconMaterial"] {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
               'Material Symbols Sharp', 'Material Icons' !important;
  font-feature-settings: 'liga';
}

/* Hide Streamlit chrome we don't want */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* --- Title rule ------------------------------------------------ */
.title-rule {
  border: 0;
  border-top: 1px solid var(--rule);
  margin: 0.4rem 0 1.6rem 0;
  width: 100%;
}
.title-sub {
  font-style: italic;
  color: var(--muted);
  font-size: 0.95rem;
  margin-top: -0.4rem;
  margin-bottom: 0.2rem;
}

/* --- Sidebar --------------------------------------------------- */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--muted);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.3rem;
  margin-top: 1.2rem;
}

/* Sidebar trace buttons — list-item style, not chunky buttons */
section[data-testid="stSidebar"] button[kind="secondary"] {
  background: transparent;
  border: 0;
  border-bottom: 1px dotted rgba(46, 37, 25, 0.35);
  border-radius: 0;
  text-align: left;
  padding: 0.55rem 0.2rem;
  color: var(--ink-soft);
  font-family: var(--serif);
  font-size: 0.92rem;
  line-height: 1.35;
  white-space: normal;
  min-height: 0;
  box-shadow: none;
  transition: background 0.12s;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
  background: var(--surface-2);
  color: var(--ink);
  border-bottom-color: var(--accent);
}

/* --- Buttons (main area) -------------------------------------- */
.stButton > button[kind="primary"],
.stForm button[kind="primaryFormSubmit"] {
  background: var(--accent);
  border: 0;
  border-radius: 1px;
  padding: 0.45rem 1.1rem;
  font-family: var(--serif);
  font-weight: 500;
  letter-spacing: 0.04em;
  box-shadow: none;
  transition: background 0.12s;
}
/* Streamlit nests a <p> inside button labels — that <p> picks up the
   broad [class*="st-"] color rule above. Force parchment on the button
   AND every descendant so the label reads as cream-on-oxblood. */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] *,
.stForm button[kind="primaryFormSubmit"],
.stForm button[kind="primaryFormSubmit"] * {
  color: var(--bg) !important;
}
.stButton > button[kind="primary"]:hover,
.stForm button[kind="primaryFormSubmit"]:hover {
  background: var(--accent-soft);
}
.stButton > button[kind="secondary"] {
  background: transparent;
  border: 1px solid var(--rule);
  border-radius: 1px;
  color: var(--ink);
  font-family: var(--serif);
  letter-spacing: 0.02em;
  box-shadow: none;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--surface);
  border-color: var(--accent);
  color: var(--accent);
}

/* --- Token row (flex-wrap container) ------------------------- */
/* The token strip is a single Streamlit container whose first child
   is a sentinel <div data-token-strip>. We use :has() to identify the
   container and flex-wrap its element-container children inline.
   Buttons inside lose all chrome and behave like ink-on-paper words. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMarkdown"] > div[data-token-strip]) {
  display: flex;
  flex-flow: row wrap;
  align-items: baseline;
  gap: 0.05rem 0.05rem;
  margin: 0.4rem 0 1.0rem 0;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMarkdown"] > div[data-token-strip]) > [data-testid="stElementContainer"] {
  width: auto !important;
  flex: 0 0 auto;
}
/* Tokens themselves: paper-like inline words */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMarkdown"] > div[data-token-strip]) .stButton > button {
  background: transparent;
  border: 0;
  border-bottom: 1px solid transparent;
  border-radius: 0;
  padding: 0 0.18rem;
  margin: 0;
  font-family: var(--serif);
  font-size: 1.3rem;
  font-weight: 400;
  color: var(--ink);
  min-height: 0;
  line-height: 1.55;
  box-shadow: none;
  transition: color 0.12s, border-color 0.12s;
  white-space: nowrap;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMarkdown"] > div[data-token-strip]) .stButton > button:hover {
  color: var(--accent);
  border-bottom-color: var(--accent-soft);
}
/* Hide the sentinel marker itself */
div[data-token-strip] { display: none; }

/* --- Translation block --------------------------------------- */
.translation-line {
  font-family: var(--serif);
  font-size: 1.55rem;
  line-height: 1.45;
  color: var(--ink);
  margin: 0.6rem 0 0.2rem 0;
}
.translation-warn {
  font-family: var(--serif);
  font-style: italic;
  color: var(--accent);
  background: var(--surface);
  padding: 0.55rem 0.85rem;
  border-left: 3px solid var(--accent);
  margin: 0.6rem 0;
}
.translation-meta {
  color: var(--muted);
  font-size: 0.82rem;
  font-family: var(--mono);
  letter-spacing: 0.02em;
}

/* --- Rating row (Roman numerals) ----------------------------- */
.rating-label {
  color: var(--muted);
  font-size: 0.82rem;
  margin-right: 0.6rem;
  font-style: italic;
}
.rating-strip + div [data-testid="stHorizontalBlock"] .stButton > button {
  background: transparent;
  border: 0;
  font-family: var(--serif);
  font-size: 1.0rem;
  letter-spacing: 0.05em;
  color: var(--muted);
  padding: 0.1rem 0.4rem;
  min-height: 0;
  box-shadow: none;
}
.rating-strip + div [data-testid="stHorizontalBlock"] .stButton > button:hover {
  color: var(--accent-soft);
}
.rating-strip + div [data-testid="stHorizontalBlock"] .stButton.rating-on > button {
  color: var(--accent);
  font-weight: 600;
}

/* --- Custom morphology table --------------------------------- */
table.morph {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--serif);
  font-size: 0.97rem;
  margin-top: 0.4rem;
}
table.morph th {
  text-align: left;
  font-weight: 600;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  border-bottom: 1px solid var(--rule);
  padding: 0.45rem 0.6rem;
}
table.morph td {
  padding: 0.45rem 0.6rem;
  border-bottom: 1px dotted rgba(46, 37, 25, 0.25);
  vertical-align: top;
  color: var(--ink-soft);
}
table.morph td.tok {
  font-size: 1.05rem;
  color: var(--ink);
}
table.morph td.lemma { font-style: italic; }
table.morph td.pos,
table.morph td.feat { font-family: var(--mono); font-size: 0.86rem; color: var(--muted); }
table.morph td.idx  { font-family: var(--mono); color: var(--muted); width: 2.4rem; }
table.morph tr.selected td { background: var(--surface-2); color: var(--ink); }
table.morph tr.selected td.lemma { color: var(--ink); }
table.morph tr.selected td.pos,
table.morph tr.selected td.feat { color: var(--ink-soft); }
table.morph tr.selected td.idx-marker {
  border-left: 3px solid var(--accent);
  padding-left: calc(0.6rem - 3px);
}

/* --- Evidence cards ------------------------------------------ */
.cite {
  border-left: 2px solid var(--rule);
  padding: 0.5rem 0.9rem;
  margin: 0.55rem 0;
  background: rgba(255, 255, 255, 0.18);
}
.cite.tm        { border-left-color: var(--gold); }
.cite.grammar   { border-left-color: var(--accent); }
.cite-mark      { font-family: var(--mono); color: var(--muted); margin-right: 0.4rem; }
.cite-head      { font-weight: 600; color: var(--ink); }
.cite-meta      { color: var(--muted); font-size: 0.84rem; font-style: italic; }
.cite-latin     { font-family: var(--serif); font-style: italic; margin-top: 0.35rem; color: var(--ink); }
.cite-english   { font-family: var(--serif); margin-top: 0.15rem; color: var(--ink-soft); }
.cite-distance  { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); }

/* --- Phenomena chips ----------------------------------------- */
.chip {
  display: inline-block;
  font-family: var(--mono);
  font-size: 0.78rem;
  padding: 0.12rem 0.55rem;
  margin: 0.15rem 0.25rem 0.15rem 0;
  background: var(--surface);
  color: var(--ink);
  border: 1px solid var(--rule);
  border-radius: 1px;
}

/* --- Inputs / textarea --------------------------------------- */
.stTextArea textarea, .stTextInput input {
  background: rgba(255, 255, 255, 0.4);
  font-family: var(--serif);
  font-size: 1.05rem;
  border: 1px solid var(--rule);
  color: var(--ink);
}
.stSelectbox div[data-baseweb="select"] {
  background: rgba(255, 255, 255, 0.4);
  border-radius: 1px;
}

/* --- Tabs ---------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 0.2rem; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab"] {
  font-family: var(--serif);
  color: var(--muted);
  background: transparent;
  border-radius: 0;
  padding: 0.5rem 0.9rem;
}
.stTabs [aria-selected="true"] {
  color: var(--accent);
  border-bottom: 2px solid var(--accent);
}

/* --- Misc ---------------------------------------------------- */
hr { border: 0; border-top: 1px solid var(--rule); margin: 1.4rem 0; }
.compare-col { padding: 0 0.6rem; }
.compare-col:first-child { border-right: 1px dotted var(--rule); }
.banner-historical {
  font-family: var(--serif);
  font-style: italic;
  color: var(--muted);
  background: var(--surface);
  border-left: 3px solid var(--gold);
  padding: 0.45rem 0.85rem;
  margin: 0.6rem 0;
}
</style>
"""


# -------------------------------------------------------------------
# Routing modes — labels match what the gateway accepts
# -------------------------------------------------------------------
_ROUTING_MODES: list[tuple[str, str]] = [
    ("(gateway default)", ""),  # empty = no override
    ("Hybrid (Claude + local)", "hybrid"),
    ("Claude only", "claude-only"),
    ("Local only (Qwen MLX)", "local-only"),
    ("DeepSeek only", "deepseek-only"),
]


# -------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------
def call_translate(text: str, routing_mode: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": text}
    if routing_mode:
        payload["routing_mode"] = routing_mode
    resp = _http_client().post(f"{_gateway_url()}/translate", json=payload)
    resp.raise_for_status()
    body: dict[str, Any] = resp.json()
    return body


def patch_rating(trace_id: str, rating: int) -> dict[str, Any]:
    resp = _http_client().patch(
        f"{_gateway_url()}/traces/{trace_id}/rating",
        json={"rating": rating},
    )
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
                "user_rating": r.user_rating,
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
            "user_rating": row.user_rating,
        }


# -------------------------------------------------------------------
# Render helpers
# -------------------------------------------------------------------
_DISPLAY_FEATURE_KEYS: tuple[str, ...] = (
    "Case", "VerbForm", "Mood", "Tense", "Aspect",
    "Voice", "Number", "Gender", "Person", "Degree",
)


def _filter_features(raw: dict[str, str] | None) -> str:
    if not raw:
        return ""
    return ", ".join(f"{k}={raw[k]}" for k in _DISPLAY_FEATURE_KEYS if k in raw)


def render_title() -> None:
    st.markdown(
        '<h1 style="margin-bottom:0">MaggieAI <span style="color:var(--muted);'
        'font-weight:400;font-size:0.6em;font-style:italic">— Lexicon Workbench</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="title-rule"/>', unsafe_allow_html=True)


def render_token_strip(tokens: list[dict[str, Any]], strip_key: str) -> None:
    """Render the source text as clickable tokens. Clicking sets
    ``selected_token`` in session_state and triggers a rerun.

    Layout: a single ``st.container`` holding the buttons; CSS uses the
    ``[data-token-strip]`` sentinel + ``:has()`` to flex-wrap inline so
    tokens flow as if they were inline words. No Streamlit columns —
    columns force equal widths and 2-token sentences would split 50/50.
    Selection highlight is a one-off CSS rule keyed on Streamlit's
    ``st-key-*`` class (present since v1.31).
    """
    if not tokens:
        return
    selected = st.session_state.get("selected_token")
    sanitised_key = strip_key.replace("-", "_")
    if selected is not None:
        st.markdown(
            f'<style>'
            f'.st-key-{sanitised_key}_tok_{selected} button {{'
            f'  color: var(--accent) !important;'
            f'  border-bottom: 1.5px solid var(--accent) !important;'
            f'  font-weight: 600 !important;'
            f'}}'
            f'</style>',
            unsafe_allow_html=True,
        )
    with st.container():
        st.markdown('<div data-token-strip="1"></div>', unsafe_allow_html=True)
        for tok in tokens:
            idx = tok.get("index")
            text = tok.get("text", "")
            if st.button(
                text,
                key=f"{sanitised_key}_tok_{idx}",
                help=f"token #{idx}",
            ):
                st.session_state["selected_token"] = idx
                st.rerun()


def render_translation(result: dict[str, Any]) -> None:
    translation = (result.get("translation") or "").strip()
    if translation:
        st.markdown(
            f'<div class="translation-line">{html.escape(translation)}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Empty translation: by contract (prompts/system_translator.j2
        # rule 6) the model declined the input — likely not Latin. The
        # rationale tab carries the explanation.
        st.markdown(
            '<div class="translation-warn">The model declined this input '
            '(likely not Latin). See the <em>Rationale</em> tab for the '
            'model\'s explanation.</div>',
            unsafe_allow_html=True,
        )
    iters = result.get("iterations", "?")
    trace_id = result.get("trace_id", "?")
    st.markdown(
        f'<div class="translation-meta">trace {trace_id} · iterations {iters}</div>',
        unsafe_allow_html=True,
    )


def render_rating(trace_id: str, current: int | None) -> None:
    """Five Roman-numeral marks: I II III IV V. Clicking sets the rating
    via PATCH /traces/{id}/rating; the resolved value is held in
    session_state[f'rating_{trace_id}'] so the highlight survives reruns.
    """
    state_key = f"rating_{trace_id.replace('-', '_')}"
    if state_key not in st.session_state and current is not None:
        st.session_state[state_key] = int(current)
    active = st.session_state.get(state_key)

    if active is not None:
        st.markdown(
            f'<style>'
            f'.st-key-{state_key}_btn_{active} button {{'
            f'  color: var(--accent) !important;'
            f'  font-weight: 600 !important;'
            f'}}'
            f'</style>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="rating-strip"></div>', unsafe_allow_html=True)
    cols = st.columns([1.4] + [0.6] * 5 + [6])
    cols[0].markdown(
        '<span class="rating-label">rate this translation</span>',
        unsafe_allow_html=True,
    )
    numerals = ["I", "II", "III", "IV", "V"]
    for i, numeral in enumerate(numerals, start=1):
        with cols[i]:
            if st.button(numeral, key=f"{state_key}_btn_{i}", help=f"rate {i} of 5"):
                try:
                    patch_rating(trace_id, i)
                    st.session_state[state_key] = i
                except httpx.HTTPError as exc:
                    st.toast(f"Could not save rating: {exc}", icon="⚠")
                st.rerun()


def render_rationale(result: dict[str, Any]) -> None:
    rationale = result.get("rationale") or "_(no rationale)_"
    st.markdown(rationale)


def render_morphology(tokens: list[dict[str, Any]], selected_idx: int | None) -> None:
    if not tokens:
        st.markdown('<p style="color:var(--muted);font-style:italic">'
                    '(no morphology data)</p>', unsafe_allow_html=True)
        return
    rows_html: list[str] = [
        '<table class="morph">',
        '<thead><tr><th>#</th><th>Token</th><th>Lemma</th>'
        '<th>POS</th><th>Features</th></tr></thead><tbody>',
    ]
    for tok in tokens:
        idx = tok.get("index", "")
        is_sel = (idx == selected_idx)
        cls = ' class="selected"' if is_sel else ''
        idx_cls = "idx idx-marker" if is_sel else "idx"
        rows_html.append(
            f'<tr{cls}>'
            f'<td class="{idx_cls}">{html.escape(str(idx))}</td>'
            f'<td class="tok">{html.escape(tok.get("text", ""))}</td>'
            f'<td class="lemma">{html.escape(tok.get("lemma") or "?")}</td>'
            f'<td class="pos">{html.escape(tok.get("pos") or "?")}</td>'
            f'<td class="feat">{html.escape(_filter_features(tok.get("features")))}</td>'
            '</tr>'
        )
    rows_html.append('</tbody></table>')
    st.markdown("\n".join(rows_html), unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--muted);font-size:0.82rem;font-style:italic;'
        'margin-top:0.5rem">Features filtered to UD canonical set; '
        'CLTK-internal fields are visible in the <em>Raw Trace</em> tab. '
        'Click a token above to highlight its row here.</p>',
        unsafe_allow_html=True,
    )


def render_evidence(result: dict[str, Any]) -> None:
    phenomena = result.get("phenomena_detected", [])
    st.markdown(
        '<h3 style="margin-top:0.4rem">Phenomena detected</h3>',
        unsafe_allow_html=True,
    )
    if phenomena:
        chips = " ".join(f'<span class="chip">{html.escape(p)}</span>' for p in phenomena)
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color:var(--muted);font-style:italic">(none)</p>',
            unsafe_allow_html=True,
        )

    citations = result.get("citations", [])
    st.markdown('<h3 style="margin-top:1.4rem">Citations</h3>', unsafe_allow_html=True)
    if not citations:
        st.markdown(
            '<p style="color:var(--muted);font-style:italic">'
            '(no citations — TM is empty and no grammar rule matched)</p>',
            unsafe_allow_html=True,
        )
        return

    for c in citations:
        kind = c.get("type")
        if kind == "translation_memory":
            author = c.get("author") or "?"
            work = c.get("work") or ""
            locator = c.get("locator") or ""
            translator = c.get("translator") or "anonymous"
            distance = c.get("distance", 0.0)
            relevance = max(0.0, 1.0 - float(distance))
            src = c.get("source_text") or ""
            tgt = c.get("target_text") or ""
            st.markdown(
                '<div class="cite tm">'
                f'<span class="cite-mark">¶</span>'
                f'<span class="cite-head">{html.escape(author)} · '
                f'{html.escape(work)} {html.escape(locator)}</span> '
                f'<span class="cite-meta">tr. {html.escape(translator)}</span>'
                f'<div class="cite-latin">{html.escape(src)}</div>'
                f'<div class="cite-english">{html.escape(tgt)}</div>'
                f'<div class="cite-distance">cosine distance {distance:.3f} · '
                f'relevance ≈ {relevance:.2f}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif kind == "grammar_rule":
            rule = c.get("rule", "?")
            source = c.get("source", "?")
            st.markdown(
                '<div class="cite grammar">'
                f'<span class="cite-mark">§</span>'
                f'<span class="cite-head">{html.escape(rule)}</span> '
                f'<span class="cite-meta">{html.escape(str(source))}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="cite">{html.escape(str(c))}</div>',
                        unsafe_allow_html=True)


def render_raw_trace(payload: dict[str, Any]) -> None:
    st.json(payload, expanded=False)


# -------------------------------------------------------------------
# Compare-mode renderer
# -------------------------------------------------------------------
def render_compare(left: dict[str, Any], right: dict[str, Any]) -> None:
    cols = st.columns(2, gap="medium")
    for col, trace in zip(cols, (left, right), strict=False):
        with col:
            st.markdown(
                f'<div class="banner-historical">'
                f'<strong>{html.escape(trace["created_at"])}</strong> · '
                f'<span class="mono">{html.escape(trace["trace_id"][:8])}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-style:italic;color:var(--ink-soft);'
                f'margin-bottom:0.4rem">{html.escape(trace["input_text"])}</div>',
                unsafe_allow_html=True,
            )
            dump = trace.get("state_dump") or {}
            output = dump.get("output") if isinstance(dump, dict) else None
            if not isinstance(output, dict):
                st.markdown(
                    '<p style="color:var(--muted);font-style:italic">'
                    '(no output stored)</p>',
                    unsafe_allow_html=True,
                )
                continue
            translation = (output.get("translation") or "").strip() or "—"
            st.markdown(
                f'<div class="translation-line" style="font-size:1.25rem">'
                f'{html.escape(translation)}</div>',
                unsafe_allow_html=True,
            )
            iters = output.get("iterations", "?")
            st.markdown(
                f'<div class="translation-meta">iterations {iters}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Rationale", expanded=False):
                st.markdown(output.get("rationale") or "_(none)_")
            with st.expander("Morphology", expanded=False):
                render_morphology(output.get("morpho_analysis") or [], None)


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
def render_sidebar(traces: list[dict[str, Any]]) -> None:
    with st.sidebar:
        st.markdown("### Routing")
        labels = [m[0] for m in _ROUTING_MODES]
        cur_label = st.session_state.get("routing_mode_label", labels[0])
        cur_idx = labels.index(cur_label) if cur_label in labels else 0
        choice = st.selectbox(
            "Inference mode",
            labels,
            index=cur_idx,
            label_visibility="collapsed",
        )
        st.session_state["routing_mode_label"] = choice
        st.session_state["routing_mode"] = dict(_ROUTING_MODES)[choice] or None

        st.markdown("### Compare")
        compare_on = st.toggle(
            "Side-by-side mode",
            value=st.session_state.get("compare_mode", False),
        )
        st.session_state["compare_mode"] = compare_on
        if compare_on:
            options = {
                f"{t['created_at']} — {t['input_text'][:40]}": t["trace_id"]
                for t in traces
            }
            picks = st.multiselect(
                "Pick exactly two traces",
                list(options.keys()),
                max_selections=2,
                default=[],
            )
            st.session_state["compare_pair"] = [options[p] for p in picks] if len(picks) == 2 else None

        st.markdown("### Recent traces")
        if not traces:
            st.markdown(
                '<p style="color:var(--muted);font-style:italic">'
                '(no traces yet — run a translation)</p>',
                unsafe_allow_html=True,
            )
            return
        for t in traces:
            preview = t["input_text"][:48] + ("…" if len(t["input_text"]) > 48 else "")
            rating_marker = ""
            if t.get("user_rating"):
                rating_marker = " · " + "I" * int(t["user_rating"])
            label = f"{t['created_at']}{rating_marker}\n\n{preview}"
            if st.button(label, key=f"trace_btn_{t['trace_id']}", width="stretch"):
                st.session_state["loaded_trace"] = t
                st.session_state.pop("last_result", None)
                st.session_state.pop("selected_token", None)
                st.rerun()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="MaggieAI — Lexicon Workbench",
        page_icon=None,
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    render_title()

    try:
        traces = fetch_recent_traces(limit=20)
    except Exception as exc:  # pragma: no cover — surface to UI
        st.error(f"Database unreachable: {exc}")
        traces = []

    render_sidebar(traces)

    # ---- Compare mode short-circuits the main panel ----------
    if st.session_state.get("compare_mode"):
        pair = st.session_state.get("compare_pair")
        if not pair:
            st.markdown(
                '<p style="color:var(--muted);font-style:italic">'
                'Pick two traces in the sidebar to compare them side-by-side.</p>',
                unsafe_allow_html=True,
            )
            return
        left = fetch_trace_by_id(pair[0])
        right = fetch_trace_by_id(pair[1])
        if not (left and right):
            st.error("One of the selected traces could not be loaded.")
            return
        render_compare(left, right)
        return

    # ---- Normal mode -----------------------------------------
    text_value = st.session_state.get("input_text", "")
    text = st.text_area(
        "Latin sentence",
        value=text_value,
        height=100,
        placeholder="Caesare imperante, Galli rebellaverunt",
        label_visibility="collapsed",
    )

    btn_cols = st.columns([1, 1, 6])
    if btn_cols[0].button("Translate", type="primary"):
        if not text.strip():
            st.warning("Enter a sentence first.")
        else:
            st.session_state["input_text"] = text
            st.session_state.pop("selected_token", None)
            mode = st.session_state.get("routing_mode")
            label = st.session_state.get("routing_mode_label", "")
            with st.spinner(f"Calling /translate ({label}) — Claude/local + morphology + retrieve…"):
                try:
                    fresh_result = call_translate(text, mode)
                    st.session_state["last_result"] = fresh_result
                    st.session_state.pop("loaded_trace", None)
                except httpx.HTTPError as exc:
                    st.error(f"Gateway error: {exc}")
                    return
    if btn_cols[1].button("Clear"):
        for k in ("last_result", "loaded_trace", "input_text", "selected_token"):
            st.session_state.pop(k, None)
        st.rerun()

    # ---- Render result, either fresh or loaded --------------
    result: dict[str, Any] | None = None
    state_dump: dict[str, Any] | None = None
    historical = False
    historical_rating: int | None = None
    historical_when = ""

    if "loaded_trace" in st.session_state:
        trace = st.session_state["loaded_trace"]
        raw_dump = trace.get("state_dump")
        if isinstance(raw_dump, dict):
            state_dump = raw_dump
            raw_output = state_dump.get("output")
            if isinstance(raw_output, dict):
                result = raw_output
        historical = True
        historical_rating = trace.get("user_rating")
        historical_when = trace.get("created_at", "")
    elif "last_result" in st.session_state:
        last = st.session_state["last_result"]
        if isinstance(last, dict):
            result = last

    if not result:
        st.markdown(
            '<p style="color:var(--muted);font-style:italic">'
            '(translate something or load a trace from the sidebar)</p>',
            unsafe_allow_html=True,
        )
        return

    if historical:
        st.markdown(
            f'<div class="banner-historical">Viewing historical trace from '
            f'<strong>{html.escape(historical_when)}</strong></div>',
            unsafe_allow_html=True,
        )

    morpho = result.get("morpho_analysis", [])
    selected_idx = st.session_state.get("selected_token")

    # Source text rendered as clickable tokens
    st.markdown("---")
    render_token_strip(morpho, strip_key=result.get("trace_id", "x"))

    # Translation + rating
    render_translation(result)
    trace_id = result.get("trace_id")
    if trace_id:
        render_rating(trace_id, historical_rating if historical else None)

    # Tabs
    rationale_tab, morph_tab, evidence_tab, trace_tab = st.tabs(
        ["Rationale", "Morphology", "Evidence", "Raw Trace"]
    )
    with rationale_tab:
        render_rationale(result)
    with morph_tab:
        render_morphology(morpho, selected_idx)
    with evidence_tab:
        render_evidence(result)
    with trace_tab:
        render_raw_trace(state_dump if state_dump is not None else result)


main()
