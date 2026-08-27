"""Advanced LangGraph orchestration with verification and bounded recovery."""
from __future__ import annotations
from typing import Any, Awaitable, Callable
from .state import GraphState

Node = Callable[[GraphState], Awaitable[dict[str, Any]]]


def build_weather_graph(
    *, router: Node, planner: Node, reasoner: Node, executor: Node,
    verifier: Node, synthesizer: Node, max_rounds: int = 4, max_retries: int = 1,
):
    """Compile Router -> Planner -> Reasoner <-> MCP -> Verify -> Synthesize.

    Verification can send the graph back through the reasoner for one bounded
    corrective retrieval cycle; this prevents uncontrolled agent loops.
    """
    if max_rounds < 1 or max_retries < 0:
        raise ValueError("max_rounds must be >= 1 and max_retries must be >= 0")
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is required. Install it with: pip install langgraph") from exc

    graph = StateGraph(GraphState)
    for name, node in (("router", router), ("planner", planner), ("reasoner", reasoner),
                       ("executor", executor), ("verifier", verifier), ("synthesizer", synthesizer)):
        graph.add_node(name, node)
    graph.add_edge(START, "router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "reasoner")

    def after_reasoner(state: GraphState) -> str:
        if state.get("next_action") == "tool" and int(state.get("rounds", 0)) < max_rounds:
            return "executor"
        return "verifier"

    def after_verifier(state: GraphState) -> str:
        verification = state.get("verification") or {}
        if verification.get("sufficient"):
            return "synthesizer"
        if int(state.get("retry_count", 0)) < max_retries and int(state.get("rounds", 0)) < max_rounds:
            return "reasoner"
        return "synthesizer"

    graph.add_conditional_edges("reasoner", after_reasoner, {"executor": "executor", "verifier": "verifier"})
    graph.add_edge("executor", "reasoner")
    graph.add_conditional_edges("verifier", after_verifier, {"reasoner": "reasoner", "synthesizer": "synthesizer"})
    graph.add_edge("synthesizer", END)
    return graph.compile()
