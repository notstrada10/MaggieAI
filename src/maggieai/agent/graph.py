"""Construction of the LangGraph StateGraph.

Graph shape:

    START
      └─> morpho_parse
            └─> phenomena_detect
                  └─> retrieve
                        └─> draft_translation
                              └─> self_critique
                                    ├─(issues + iter < MAX)─> draft_translation  (loop)
                                    └─(else)──────────────> format_output
                                                                  └─> END
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from maggieai.agent import nodes
from maggieai.agent.state import AgentState
from maggieai.inference.router import InferenceRouter


def build_graph(router: InferenceRouter | None = None) -> Any:
    """Build and compile the graph. Inject `router` for testability."""
    router = router or InferenceRouter()

    g: StateGraph[AgentState] = StateGraph(AgentState)

    g.add_node("morpho_parse", nodes.morpho_parse)
    g.add_node("phenomena_detect", nodes.phenomena_detect)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("draft_translation", partial(nodes.draft_translation, router=router))
    g.add_node("self_critique", partial(nodes.self_critique, router=router))
    g.add_node("format_output", nodes.format_output)

    g.add_edge(START, "morpho_parse")
    g.add_edge("morpho_parse", "phenomena_detect")
    g.add_edge("phenomena_detect", "retrieve")
    g.add_edge("retrieve", "draft_translation")
    g.add_edge("draft_translation", "self_critique")
    g.add_conditional_edges(
        "self_critique",
        nodes.should_iterate,
        {"draft_translation": "draft_translation", "format_output": "format_output"},
    )
    g.add_edge("format_output", END)

    return g.compile()
