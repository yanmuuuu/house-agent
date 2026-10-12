from langgraph.graph import MessagesState

class ReserverState(MessagesState):
    title: str          # 预定的房源
    phone: str          # 电话号
    IdCard: str         # 身份证

