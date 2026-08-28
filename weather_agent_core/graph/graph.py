"""Advanced LangGraph orchestration with verification and bounded recovery."""
from __future__ import annotations
from typing import Any, Awaitable, Callable
from .state import GraphState

Node = Callable[[GraphState], Awaitable[dict[str, Any]]]


async def _default_verifier(_: GraphState) -> dict[str, Any]:
    """Compatibility verifier for callers that do not need verification."""
    return {"verification": {"sufficient": True, "reason": "verification not configured"}}


def build_weather_graph(
    *, router: Node, planner: Node, reasoner: Node, executor: Node,
    synthesizer: Node, verifier: Node | None = None,
    max_rounds: int = 4, max_retries: int = 1,
):
    """Compile Router -> Planner -> Reasoner <-> MCP -> Verify -> Synthesize.

    The executor goes directly to verification once all required plan groups are
    satisfied. This prevents an unnecessary Gemini round after a successful tool
    batch while retaining the reasoner loop when required evidence is missing.
    """
    if max_rounds < 1 or max_retries < 0:
        raise ValueError("max_rounds must be >= 1 and max_retries must be >= 0")
    if verifier is None:
        verifier = _default_verifier
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

    def after_executor(state: GraphState) -> str:
        """Verify immediately when the current batch already covers the plan."""
        plan = state.get("plan") or {}
        observations = state.get("observations") or []
        successful = {
            str(item.get("tool"))
            for item in observations
            if item.get("tool")
            and not (isinstance(item.get("result"), dict) and item["result"].get("success") is False)
        }
        required_groups = [
            set(step.get("preferred_tools", []))
            for step in plan.get("steps", [])
            if step.get("required", True)
        ]
        satisfied = all(not group or group.intersection(successful) for group in required_groups)
        return "verifier" if satisfied else "reasoner"

    def after_verifier(state: GraphState) -> str:
        verification = state.get("verification") or {}
        if verification.get("sufficient"):
            return "synthesizer"
        retry_count = int(state.get("retry_count", 0))
        if retry_count <= max_retries and int(state.get("rounds", 0)) < max_rounds:
            return "reasoner"
        return "synthesizer"

    graph.add_conditional_edges("reasoner", after_reasoner, {"executor": "executor", "verifier": "verifier"})
    graph.add_conditional_edges("executor", after_executor, {"reasoner": "reasoner", "verifier": "verifier"})
    graph.add_conditional_edges("verifier", after_verifier, {"reasoner": "reasoner", "synthesizer": "synthesizer"})
    graph.add_edge("synthesizer", END)
    return graph.compile()
