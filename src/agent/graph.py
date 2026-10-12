from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agent.recommend import recommend_graph
from agent.extend import extend_graph
from agent.reserve import reserve_graph
from agent.common.context import ContextSchema
from agent.node.main import get_store_info, intent_recognition, need_reserve, get_user_preferences
from agent.state.main import State, NeedReserveOutput

builder = StateGraph(State, context_schema=ContextSchema)

builder.add_node(get_store_info)
builder.add_node(intent_recognition)
builder.add_node(need_reserve)
builder.add_node(get_user_preferences)
builder.add_node("recommend_graph", recommend_graph)
builder.add_node("reserve_graph", reserve_graph)
builder.add_node("extend_graph", extend_graph)

builder.add_edge(START, "get_store_info")
builder.add_edge("get_store_info", "intent_recognition")
def smart_router(state: State):
    if state["user_intent"] == "recommend_house":
        return "recommend_graph"
    elif state["user_intent"] == "reserve_house":
        return "reserve_graph"
    elif state["user_intent"] == "get_info":
        return "get_user_preferences"
    else:
        return "extend_graph"
builder.add_conditional_edges(
    "intent_recognition",
    smart_router,
    ["recommend_graph", "reserve_graph", "get_user_preferences", "extend_graph"],
)
builder.add_edge("get_user_preferences", END)
builder.add_edge("extend_graph", END)
builder.add_edge("reserve_graph", END)
builder.add_edge("recommend_graph", "need_reserve")


def should_reserve(state: NeedReserveOutput):
    if state["is_reserve"] == "需要":
        return "reserve_graph"
    else:
        return END


builder.add_conditional_edges(
    "need_reserve",
    should_reserve,
    [END, "reserve_graph"],
)

graph = builder.compile()
