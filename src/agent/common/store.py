from typing import Optional
from pydantic import Field, BaseModel

class ReservedInfo(BaseModel):
    """房源的预定信息"""
    order_id: str = Field(default=None, description="预定id")
    title: str = Field(default=None, description="房源标题")
    phone_number: str = Field(default=None, description="预定电话")
    price: Optional[float] = Field(default=None, description="预定的房源价格(元/月)")
    intro: Optional[str] = Field(default=None, description="预定的房源介绍")
    city_name: Optional[str] = Field(default=None, description="预定的房源所在城市名称")
    region_name: Optional[str] = Field(default=None, description="预定的房源所在区/县")

class UserPreferences(BaseModel):
    """用户偏好"""
    budget_min: Optional[float] = Field(default=None, description="用户的最低预算，单位为元/月")
    budget_max: Optional[float] = Field(default=None,description="用户的最高预算，单位为元/月")
    reserved_info: Optional[list[ReservedInfo]] = Field(default=None,description="预定过的房源列表")