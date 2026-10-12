from langchain_core.messages import SystemMessage, filter_messages, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import Literal

from agent.common.context import ContextSchema
from agent.common.llm import model
from agent.state.main import State
from agent.state.main import NeedReserveOutput

"""-- 获取用户偏好数据 --"""
def get_store_info(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore):
    user_id = runtime.context["user_id"]
    namespace = (user_id, "Preference")
    list_searchItem = store.search(namespace)
    if list_searchItem and list_searchItem[0]:
        return {
            "user_preferences": list_searchItem[0].value
        }
    else:
        return {
            "user_preferences": {}
        }

"""-- 意图识别 --"""
class recognition(BaseModel):
    intent: Literal["recommend_house", "reserve_house", "get_info", "others"] \
            = Field(description="根据用户问题描述判断问题类型: 推荐房源, 预定房源, 获取信息, 其他内容")
def intent_recognition(state: State):
    result = model.with_structured_output(recognition).invoke(
        [SystemMessage(content="你是一个根据描述提取信息的提取专家,请从用户的描述中提取想要咨询的相关信息, 严谨根据语义推断信息, 但是不能猜测或者编造信息")
        , state["messages"][-1]])
    return {
        "user_intent": result.intent
    }

"""-- 是否需要预约 --"""
def need_reserve(state: State) -> NeedReserveOutput:
    prompt = f"已经为您推荐合适的房源, 是否需要帮您预订房源?\n"
    prompt += "如果不需要要，请输⼊'**不需要**'。\n"
    prompt += "如果需要，请输⼊'**需要**'。\n(注意输⼊其它值无效)"
    # 中断，等待用户输⼊
    answer = interrupt(prompt)
    if answer == "需要":
        return {
            "is_reserve": "需要"
        }
    else:
        return {
            "is_reserve": "不需要"
        }

"""-- 返回用户详细偏好信息(根据用户消息+历史偏好 -> 返回ai总结的用户信息) --"""
def get_user_preferences(state: State):
    prefs = state.get("user_preferences", {})
    human_messages = filter_messages(state["messages"], include_types=["human"])
    list_reserved_info = prefs.get("reserved_info",[])
    reserved_info_str = ""
    if list_reserved_info:
        for i, item in enumerate(list_reserved_info, 1):
            reserved_info_str += f"{i}.[预约工单号:{item.get('order_id')}, 房源标题:{item.get('title')}, 预约电话:{item.get('phone_number')}]"
    result = model.invoke([SystemMessage(content="你是⼀个乐于助⼈的助手，可以根据用户偏好信息进⾏回复。"
                                       "如果有的偏好数据为空，不要猜测或编造数据。"
                                       "不要直接回复偏好数据是什么，要结合问题进行生动回复。"
                                       "如果问题与用户偏好数据无关，直接回复即可")
                 , HumanMessage(content=f"用户偏好数据如下: 最低预算(元/月): {prefs.get('budget_min')}, 最高预算(元/月): {prefs.get('budget_max')}, 预约记录: {reserved_info_str}")
                 , human_messages[-1]])
    return {
        "messages": [result]
    }