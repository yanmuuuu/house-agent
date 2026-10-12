import os
import re
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import filter_messages, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.constants import END
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent.common.context import ContextSchema
from agent.common.llm import model
from agent.common.store import UserPreferences
from agent.state.recommend import RecommendState, get_recommend_info


class UserInfo(BaseModel):
    city: Optional[str] = Field(default=None, description="用户所在或想要租房的城市，例如：西安、北京、上海") # 城市
    budget_min: Optional[float] = Field(
        default=None,
        description="最低预算(元/月)。例如「3000以上/不低于3000/3000起/至少3000」时填该数字，budget_max 留空；"
                    "「3000以内/以下/不超过」时不要填本字段。"
    ) # 最低预算
    budget_max: Optional[float] = Field(
        default=None,
        description="最高预算(元/月)。例如「3000以内/以下/不超过/低于3000」时填该数字，budget_min 留空；"
                    "「3000以上」时不要填本字段。"
    ) # 最⾼预算
    district: Optional[str] = Field(default=None, description="用户想要租房的具体区域或⾏政区，例如：雁塔区、碑林区、海淀区") # 区域
    room_type: Optional[str] = Field(default=None, description="房屋类型，例如：整租、合租、公寓、⼀室⼀厅、两室⼀厅") # 房屋类型
    orientation: Optional[str] = Field(default=None, description="房屋朝向，例如：朝南、朝北、东南、南北通透") # 朝向
    room_count: Optional[int] = Field(default=None, description="需要推荐的房屋数量") # 推荐数量
    others: Optional[str] = Field(default=None, description="特殊要求，例如：带阳台、独⽴卫⽣间、近地铁、可养宠物、有电梯等") # 其它要求


_RANGE_BUDGET_RE = re.compile(r"\d+(?:\.\d+)?\s*[-~到至]\s*\d+")
_UPPER_BUDGET_RE = re.compile(
    r"(?:不超过|不高于|(?<!不)(?:低于|不到|少于))\s*(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*(?:块钱|块|元)?\s*(?:以内|以下|之内|封顶)"
)
_LOWER_BUDGET_RE = re.compile(
    r"(?:不低于|至少|(?<!不)(?:高于|多于|超过))\s*(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*(?:块钱|块|元)?\s*(?:以上|起)"
)


def _first_number(match: Optional[re.Match]) -> Optional[float]:
    if match is None:
        return None
    for group in match.groups():
        if group is not None:
            return float(group)
    return None


def parse_fuzzy_budget(text: str) -> tuple[Optional[float], Optional[float]]:
    """从「3000以内 / 3000以上」这类话术里解析单侧预算；明确区间交给模型提取。"""
    if not text or _RANGE_BUDGET_RE.search(text):
        return None, None
    return _first_number(_LOWER_BUDGET_RE.search(text)), _first_number(_UPPER_BUDGET_RE.search(text))


def fill_fuzzy_budget(
    info: UserInfo,
    text: str = "",
    hist_min: Optional[float] = None,
    hist_max: Optional[float] = None,
) -> UserInfo:
    """模型只抽出一侧、或完全没抽出时，按模糊预算规则补全 min/max。"""
    if info is None:
        info = UserInfo()
    parsed_min, parsed_max = parse_fuzzy_budget(text)
    if info.budget_min is None and parsed_min is not None:
        info.budget_min = parsed_min
    if info.budget_max is None and parsed_max is not None:
        info.budget_max = parsed_max
    if info.budget_max is not None and info.budget_min is None:
        candidate = hist_min if hist_min is not None else 0.0
        info.budget_min = candidate if candidate <= info.budget_max else 0.0
    elif info.budget_min is not None and info.budget_max is None:
        if hist_max is not None and hist_max >= info.budget_min:
            info.budget_max = hist_max
        else:
            info.budget_max = info.budget_min + 10000.0
    return info

"""-- 收集用户信息的节点 --"""
def collect_user_info(state: RecommendState, runtime: Runtime[ContextSchema], *, store: BaseStore):
    user_preferences = state.get("user_preferences") # 获取用户偏好
    human_messages = filter_messages(messages=state["messages"], include_types="human") # 用户消息
    extract_messages = []
    if user_preferences:
        if user_preferences.get("budget_max"):
            extract_messages += [HumanMessage(content=f"下面是用户的历史偏好数据->最大预算{user_preferences['budget_max']}元/月")]
        if user_preferences.get("budget_min"):
            extract_messages += [HumanMessage(content=f"下面是用户的历史偏好数据->最低预算{user_preferences['budget_min']}元/月")]
    extract_messages += [human_messages[-1]]

    def extract_info_by_messages(messages) -> UserInfo:
        system_messages = SystemMessage(content="你是⼀个租房需求信息提取专家。请从用户的描述与历史信息中提取租房相关信息。"
                                                "如果用户历史偏好信息与最新用户消息冲突，以最新的用户消息为主。"
                                                "只提取用户明确提到的信息，不要猜测或推断。"
                                                "如果某个信息用户没有提到，就返回None。"
                                                "注意预算的单位可能是元/月、元/天等，请统⼀转换为元/月。"
                                                "模糊预算必须提取，不要因为只说了一侧价格就返回None："
                                                "1) 「3000以内/以下/不超过/低于3000」：只填 budget_max=3000.0，budget_min 留空为None。"
                                                "2) 「3000以上/不低于3000/3000起/至少3000」：只填 budget_min=3000.0，budget_max 留空为None。"
                                                "3) 「2000到3000 / 2000-3000」：budget_min=2000.0，budget_max=3000.0。"
                                                "如果用户提到推荐几套房，提取room_count字段")
        return model.with_structured_output(UserInfo).invoke([system_messages] + messages)
    def update_state_by_extract(current_state: dict , info: UserInfo) -> dict:
        if info is None:
            return current_state
        info_dict = info.model_dump(exclude_none=True)
        current_state.update(info_dict)
        return current_state

    hist_min = user_preferences.get("budget_min") if user_preferences else None
    hist_max = user_preferences.get("budget_max") if user_preferences else None
    latest_text = human_messages[-1].content if human_messages else ""
    if not isinstance(latest_text, str):
        latest_text = str(latest_text)
    update_state = {}
    user_info = fill_fuzzy_budget(
        extract_info_by_messages(extract_messages),
        latest_text,
        hist_min,
        hist_max,
    )
    update_state = update_state_by_extract(update_state, user_info)
    missing_info = [] # 缺失的信息
    if not update_state.get("city"):
        missing_info.append("**城市**")
    if update_state.get("budget_min") is None or update_state.get("budget_max") is None:
        missing_info.append("**预算**")
    if missing_info:
        prompt = f"为了给您推荐合适的房源，请提供以下信息：{', '.join(missing_info)}和其它信息。\n"
        prompt += "如果您不想提供，请输入'**不提供**'，我会根据已有信息为您推荐房源。"
        answer = interrupt(prompt) # 中断
        if answer == "不提供":
            if not update_state.get("city"):
                update_state["city"] = "任意城市"
            if update_state.get("budget_min") is None:
                update_state["budget_min"] = 0.0
            if update_state.get("budget_max") is None:
                update_state["budget_max"] = 100000.0
            if not update_state.get("room_count"):
                update_state["room_count"] = 5
        else:
            answer_text = str(answer)
            user_info = fill_fuzzy_budget(
                extract_info_by_messages([HumanMessage(content=answer_text)]),
                answer_text,
                update_state.get("budget_min", hist_min),
                update_state.get("budget_max", hist_max),
            )
            update_state_by_extract(update_state, user_info)

    # 持久化（LangGraph Studio 未配置 context 时 runtime.context 可能为 None）
    if update_state.get("budget_min") is not None or update_state.get("budget_max") is not None:
        user_id = runtime.context["user_id"]
        namespace = (user_id, "Preference")
        search_item = store.search(namespace)
        if len(search_item) == 0:
            pref = UserPreferences(
                budget_min=update_state.get("budget_min"),
                budget_max=update_state.get("budget_max")
            )
            store.put(
                namespace=namespace,
                key=uuid.uuid4(),
                value=pref.model_dump(exclude_none=True)
            )
            update_state["user_preferences"] = pref.model_dump(exclude_none=True)
        else:
            pref = search_item[0].value
            if update_state.get("budget_min") is not None:
                pref["budget_min"] = update_state["budget_min"]
            if update_state.get("budget_max") is not None:
                pref["budget_max"] = update_state["budget_max"]
            store.put(
                namespace=namespace,
                key=search_item[0].key,
                value=pref
            )
            update_state["user_preferences"] = pref

    # 生成答案
    update_state["messages"] = [HumanMessage(content=get_recommend_info(update_state))]

    return update_state

# 配置连接数据库
load_dotenv()
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# 获取数据库工具包
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
# for tool in tools:
#     print(tool.name)
# sql_db_query
# sql_db_schema
# sql_db_list_tables
# sql_db_query_checker

# 获取表信息的工具节点
get_schema_tool =  next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

# 执行sql查询的工具节点
run_query_tool =  next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

"""-- 查询表的节点 --"""
def list_tables(state: RecommendState):
    # AiMessages(tool_call) -> ToolMessages -> AiMessages
    # 1.手动带上tool_call(而不是需要模型自己判断调用)
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "114514",
        "type": "tool_call"
    }
    tool_call_messages = AIMessage(content="", tool_calls=[tool_call])
    # 2.工具返回
    list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
    tool_messages = list_tables_tool.invoke(tool_call)
    response = AIMessage(content=f"下面是可用的表: {tool_messages.content}")
    return {
        "messages": [tool_call_messages, tool_messages, response]
    }

"""-- 调用获取详细表信息的节点(传出AiMessage, 包含tool_call, 外部调用get_schema_node) --"""
def call_get_schema(state: RecommendState):
    model_bind_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = model_bind_tools.invoke(state["messages"])
    return {
        "messages": [response]
    }

"""-- 生成sql查询语句/总结sql查询结果的节点 --"""
def generate_query(state: RecommendState):
    generate_query_system_prompt = """
    您是一个设计用于与SQL数据库交互的代理.
    给定一个输入问题，创建一个语法正确的{dialect}查询来运行，然后查看查询的结果并返回答案.
    需要根据rows from table的示例设置真实查询的值.
    除非用户指定了他们希望获得的特定数量的示例，否则始终将查询限制为最多{top_k}个结果.
    您可以按相关列对结果排序，以返回最感兴趣的结果.
    不要查询特定表中的所有列，只查询给定问题的相关列.
    不要对数据库做任何DML语句（INSERT， UPDATE， DELETE， DROP等)."""
    system_prompt = generate_query_system_prompt.format(
        dialect=db.dialect,
        top_k=state.get("room_count", 5)
    )
    system_message = SystemMessage(content=system_prompt)
    # 在这里没有强制工具调用，以允许模型在获得解决方案时自然响应。
    model_bind_tools = model.bind_tools([run_query_tool])
    # 将用户信息也加入到查询条件中
    response = model_bind_tools.invoke([system_message] + state["messages"])
    return {
        "messages": [response]
    }

"""-- 检查SQL查询语句的节点 --"""
def check_query(state: RecommendState):
    check_query_system_prompt = """
      你是一个非常注重细节的SQL专家。仔细检查{dialect}查询中的常见错误，包括:
      -使用NULL值的NOT IN
      -在应该使用UNION ALL时使用UNION
      -使用BETWEEN表示独占范围
      -谓词中的数据类型不匹配
      -正确引用标识符
      -使用正确数量的函数参数
      -转换为正确的数据类型
      -使用合适的列进行连接
      如果存在上述任何错误，请重写查询。如果没有错误，只需复制原始查询即可。
      在运行此检查之后，您将调用适当的工具来执行查询。"""
    system_prompt = check_query_system_prompt.format(
        dialect=db.dialect
    )
    system_message = SystemMessage(content=system_prompt)
    # 生成人工用户消息进行检查
    # 上一个节点是generate_query.如果走到这, 必定调用了工具, 这样获取到的SQL是准确的
    tool_call = state["messages"][-1].tool_calls[0]
    # 将SQL当作用户消息传入进行检查
    user_message = HumanMessage(content=tool_call["args"]["query"])
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {
        "messages": [response]
    }

