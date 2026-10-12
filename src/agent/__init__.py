"""New LangGraph Agent.

This module defines a custom graph.
"""
from agent.extend import extend_graph
from agent.graph import graph
from agent.recommend import recommend_graph
from agent.reserve import reserve_graph

__all__ = ["graph", "recommend_graph", "reserve_graph", "extend_graph"]
