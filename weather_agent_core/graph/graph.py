"""LangGraph orchestration graph for the weather agent."""
from __future__ import annotations
from typing import Any, Awaitable, Callable
from .state import GraphState

Node = Callable[[GraphState], Awaitable[dict[str, Any]]]


def build_weather_graph(*, router: Node, planner: Node, reasoner: Node, executor: Node, synthesizer: Node, max_rounds: int = 4):
    """Compile Router -> Planner -> Reasoner <-> Executor -> Synthesizer."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is required. Install it with: pip install langgraph") from exc

    graph = StateGraph(GraphState)
    graph.add_node("router", router)
    graph.add_node("planner", planner)
    graph.add_node("reasoner", reasoner)
    graph.add_node("executor", executor)
    graph.add_node("synthesizer", synthesizer)
    graph.add_edge(START, "router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "reasoner")

    def after_reasoner(state: GraphState) -> str:
        if state.get("next_action") == "tool" and int(state.get("rounds", 0)) < max_rounds:
            return "executor"
        return "synthesizer"

    graph.add_conditional_edges("reasoner", after_reasoner, {"executor": "executor", "synthesizer": "synthesizer"})
    graph.add_edge("executor", "reasoner")
    graph.add_edge("synthesizer", END)
    return graph.compile()
