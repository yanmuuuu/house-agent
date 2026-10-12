from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from agent.common.context import ContextSchema
from agent.node.reserve import get_title, get_phone, get_id, add_reserve_messages, tool_node, call_orders
from agent.state.reserve import ReserverState

builder = StateGraph(ReserverState, context_schema=ContextSchema)

builder.add_node(get_title)
builder.add_node(get_phone)
builder.add_node(get_id)
builder.add_node(add_reserve_messages)
builder.add_node("tool_node", tool_node)
builder.add_node(call_orders)

builder.add_edge(START, "get_title")
builder.add_edge("get_title", "get_phone")
builder.add_edge("get_phone", "get_id")
builder.add_edge("get_id", "add_reserve_messages")
builder.add_edge("add_reserve_messages", "call_orders")
builder.add_conditional_edges(
    "call_orders",
    tools_condition,
    {
        "__end__": END,
        "tools": "tool_node"
    }
)
builder.add_edge("tool_node", "call_orders")

reserve_graph = builder.compile()