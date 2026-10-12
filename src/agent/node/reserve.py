import re
import uuid
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, ToolRuntime, InjectedStore
from langgraph.types import interrupt

from agent.common.llm import model
from agent.common.store import ReservedInfo, UserPreferences
from agent.state.reserve import ReserverState


def get_title(state: ReserverState):
    prompt = "请输⼊要预定的房源名称"
    while True:
        title = interrupt(prompt)
        if title:
            return {
                "title": title
            }
        # 每次验证失败后，提示信息会更新
        prompt = f"'{title}' 不是⼀个有效的房源名称，请更正。"

def get_phone(state: ReserverState):
    prompt = "请输⼊预留的电话号码"
    while True:
        phone = interrupt(prompt)
        if re.fullmatch(r'1[3-9]\d{9}', phone):
            return {
                "phone": phone
            }
        # 每次验证失败后，提示信息会更新
        prompt = f"'{phone}' 不是⼀个有效的电话号码，请更正。"

def get_id(state: ReserverState):
    prompt = "请输⼊预留的身份证号码"
    while True:
        IdCard = interrupt(prompt)
        if re.fullmatch(r'[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]', IdCard):
            return {
                "IdCard": IdCard
            }
        # 每次验证失败后，提示信息会更新
        prompt = f"'{IdCard}' 不是⼀个有效的身份证号码，请更正。"

def add_reserve_messages(state: ReserverState):
    prompt = ("根据提供的信息，帮我预定房源。"
              "- 预定的房源标题: {title}"
              "- 用户预定号码: {phone}"
              "- 用户身份证号码: {IdCard}")
    return {
        "messages": [HumanMessage(content=prompt.format(
            title=state["title"],
            phone=state["phone"],
            IdCard=state["IdCard"]
        ))]
    }

@tool
def generate_orders(phone: str, IdCard: str, title: str, runtime: ToolRuntime, store: Annotated[Any, InjectedStore()]) -> str:
    """
    根据用户预留的电话号码, 身份证号码, 预定房源并且生成工单
    Args:
        phone: 用户电话
        IdCard: 身份证
        title: 用户要预定的房源标题
        runtime: 工具的运行时信息
        store: 注入工具的持久存储
    """
    # 1. 生成工单号
    order_id = str(uuid.uuid4())

    # 2.构建预定信息
    reserved_info = ReservedInfo(
        order_id=order_id,
        title=title,
        phone_number=phone
    )

    # 3.持久化存储
    user_id = runtime.context["user_id"]
    namespace = (user_id, "Preference")
    result_list = store.search(namespace)
    if len(result_list) == 0:
        prefs = UserPreferences(
            reserved_info=[reserved_info]
        )
        store.put(
            namespace,
            str(uuid.uuid4()),
            prefs.model_dump(exclude_none=True)
        )
    else:
        prefs = result_list[0].value or {}
        prefs.setdefault("reserved_info", []).append(reserved_info.model_dump(exclude_none=True))
        store.put(
            namespace,
            result_list[0].key,
            prefs
        )
    return f"已预定房源: {title}, 预约工单号为: {order_id}"

tool_node = ToolNode([generate_orders])

def call_orders(state: ReserverState):
    model_bind_tools = model.bind_tools([generate_orders])
    response = model_bind_tools.invoke(
        [SystemMessage(content="你是⼀个工单生成的助手: 支持调用工具进⾏房源预定工单生成; 支持查看查询的结果并返回最终答案")]
        + state["messages"])
    return {
        "messages": [response]
    }