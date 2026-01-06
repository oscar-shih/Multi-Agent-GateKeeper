from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from gatekeeper.graph.state import GatekeeperState
from gatekeeper.graph.nodes import (
    load_inputs_node,
    parse_and_normalize_node,
    compute_cfl_tool_node,
    precompute_metrics_node,
    phase1_agents_node,
    needs_debate_router,
    debate_one_round_node,
    synthesize_verdict_node,
    retrieve_similar_runs_node,
)


def build_gatekeeper_graph():
    g = StateGraph(GatekeeperState)

    g.add_node("load_inputs", load_inputs_node)
    g.add_node("parse_and_normalize", parse_and_normalize_node)
    g.add_node("compute_cfl", compute_cfl_tool_node)
    g.add_node("precompute_metrics", precompute_metrics_node)
    g.add_node("retrieve_similar_runs", retrieve_similar_runs_node)
    g.add_node("phase1_agents", phase1_agents_node)
    g.add_node("debate_one_round", debate_one_round_node)
    g.add_node("synthesize_verdict", synthesize_verdict_node)

    g.add_edge(START, "load_inputs")
    g.add_edge("load_inputs", "parse_and_normalize")
    g.add_edge("parse_and_normalize", "compute_cfl")
    g.add_edge("compute_cfl", "precompute_metrics")
    g.add_edge("precompute_metrics", "retrieve_similar_runs")
    g.add_edge("retrieve_similar_runs", "phase1_agents")

    g.add_conditional_edges(
        "phase1_agents",
        needs_debate_router,
        ["debate_one_round", "synthesize_verdict"],
    )

    g.add_edge("debate_one_round", "synthesize_verdict")
    g.add_edge("synthesize_verdict", END)

    return g.compile()