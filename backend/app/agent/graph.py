from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    AgentServices,
    make_agent_retrieve_node,
    make_decompose_node,
    make_generate_node,
    make_grade_node,
    make_parallel_retrieve_node,
    make_refine_node,
    make_rewrite_node,
    make_route_node,
)
from app.agent.state import AgentState


def build_agent_graph(
    services: AgentServices,
    enable_correction: bool = False,
    checkpointer=None,
):
    """Compile the agent graph.

    route ─(direct)─────────────────────────────────────────────▶ generate
          ├(knowledge)─▶ rewrite ────▶ agent_retrieve (tool loop) ─┐
          └(multihop)──▶ decompose ──▶ parallel_retrieve ─────────┤
                                                    ┌─────────────┘
                                                    ▼
                                     ┌─(P1)─▶ grade ─┬─ sufficient ─▶ generate
                                     │              └─ insufficient ─▶ refine ─▶ agent_retrieve
                                     └─(P0)──────────────────────────▶ generate
    """
    builder = StateGraph(AgentState)
    builder.add_node("route", make_route_node(services))
    builder.add_node("rewrite", make_rewrite_node(services))
    builder.add_node("decompose", make_decompose_node(services))
    builder.add_node("agent_retrieve", make_agent_retrieve_node(services))
    builder.add_node("parallel_retrieve", make_parallel_retrieve_node(services))
    builder.add_node("generate", make_generate_node(services))

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda state: state.get("route", "knowledge"),
        {"direct": "generate", "knowledge": "rewrite", "multihop": "decompose"},
    )
    builder.add_edge("rewrite", "agent_retrieve")
    builder.add_edge("decompose", "parallel_retrieve")

    if enable_correction:
        max_iterations = services.settings.agent_max_iterations

        def _next_after_grade(state: dict) -> str:
            if state.get("grade") == "sufficient":
                return "generate"
            if state.get("iterations", 0) >= max_iterations:
                return "generate"
            return "refine"

        builder.add_node("grade", make_grade_node(services))
        builder.add_node("refine", make_refine_node(services))
        builder.add_edge("agent_retrieve", "grade")
        builder.add_edge("parallel_retrieve", "grade")
        builder.add_conditional_edges(
            "grade", _next_after_grade, {"generate": "generate", "refine": "refine"}
        )
        builder.add_edge("refine", "agent_retrieve")
    else:
        builder.add_edge("agent_retrieve", "generate")
        builder.add_edge("parallel_retrieve", "generate")

    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)
