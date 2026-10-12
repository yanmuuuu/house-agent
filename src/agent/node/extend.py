from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

from agent.common.llm import model


def extend_node(state: MessagesState):
    return {
        "messages": model.invoke([SystemMessage(content="你是一个乐于助人的ai助手, 请回答用户问题")] + state["messages"])
    }