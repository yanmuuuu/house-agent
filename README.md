# house-agent

用 [LangGraph](https://github.com/langchain-ai/langgraph) 做的租房助手。它先判断用户想做什么，再去推荐房源、预约看房，或根据已经记下的偏好回答问题。

## 能做什么

- **推荐房源**：从对话里整理城市、区域、预算、户型、朝向和其它要求，对 MySQL 里的房源库生成查询，再把结果说给人听。提示词要求只做查询，不执行插入、更新或删除。
- **预约看房**：依次确认房源名称、手机号和身份证号，生成一条预约记录，并记到该用户的偏好里。
- **查看偏好**：结合历史预算和预约记录回答，缺的字段不会编造。
- **其它问题**：走普通对话。

主入口是 `house_agent`，定义在 `src/agent/graph.py`。推荐结束后会问要不要预约；回答「需要」才会进入预约流程。

## 图

| 名称 | 入口 |
| --- | --- |
| `house_agent` | `src/agent/graph.py` |
| `recommend_agent` | `src/agent/recommend.py` |
| `reserve_agent` | `src/agent/reserve.py` |
| `extend_agent` | `src/agent/extend.py` |

调用 `house_agent` 时带上运行时上下文 `user_id`，用来区分每个用户的偏好，例如 `{"user_id": "user-1"}`。

## 准备环境

需要 Python 3.10 或更高版本，以及一套放房源数据的 MySQL。

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -e . "langgraph-cli[inmem]"
copy .env.example .env
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -e . "langgraph-cli[inmem]"
cp .env.example .env
```

然后只在本地的 `.env` 里填写下面这些值。`.env` 已被忽略，不要把它提交到仓库。

| 变量 | 作用 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek 密钥。模型在 `src/agent/common/llm.py`，当前使用 `deepseek-v4-flash`。 |
| `DB_USER` `DB_PASSWORD` `DB_HOST` `DB_PORT` `DB_NAME` | 房源 MySQL 的连接信息。 |
| `LANGSMITH_TRACING` `LANGSMITH_PROJECT` `LANGSMITH_API_KEY` | 可选。只有需要链路追踪时再打开。 |

## 本地运行

```bash
langgraph dev
```

Studio 里选择 `house_agent`，并在 context 中填写 `user_id`。推荐和预约途中会暂停并等待输入，按提示回复即可继续。

## 开发

```bash
python -m pytest tests/unit_tests
```

## 许可

MIT。本项目由 LangGraph 的 new-langgraph-project 模板改来。
