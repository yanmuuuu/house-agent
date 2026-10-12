from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

from agent.node.extend import extend_node

builder = StateGraph(MessagesState)

builder.add_node(extend_node)

builder.add_edge(START, "extend_node")
builder.add_edge("extend_node", END)

extend_graph = builder.compile()