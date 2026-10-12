from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agent.common.context import ContextSchema
from agent.node.recommend import collect_user_info, get_schema_node, run_query_node, list_tables, call_get_schema, \
    generate_query, check_query
from agent.state.recommend import RecommendState

builder = StateGraph(RecommendState, context_schema=ContextSchema)

builder.add_node(collect_user_info)
builder.add_node("get_schema_node", get_schema_node)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node("run_query_node", run_query_node)

builder.add_edge(START, "collect_user_info")
builder.add_edge("collect_user_info", "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema_node")
builder.add_edge("get_schema_node", "generate_query")
def should_continue(state: RecommendState):
    if state["messages"][-1].tool_calls:
        return "check_query"
    return END
builder.add_conditional_edges(
    "generate_query",
    should_continue,
    [END, "check_query"]
)
builder.add_edge("check_query", "run_query_node")
builder.add_edge("run_query_node", "generate_query")

recommend_graph = builder.compile()