"""高德地图服务 - LangChain 版本"""

import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.sessions import StdioConnection
from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo

# 全局MCP工具列表
_amap_tools: Optional[List[BaseTool]] = None


def get_amap_tools() -> List[BaseTool]:
    """
    获取高德地图工具列表(LangChain MCP 适配器)

    Returns:
        LangChain BaseTool 列表
    """
    global _amap_tools

    if _amap_tools is None:
        settings = get_settings()

        if not settings.amap_api_key:
            raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")

        # 使用 langchain-mcp-adapters 加载 MCP 工具
        connection = StdioConnection(
            command="uvx",
            args=["amap-mcp-server"],
            env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
            transport="stdio"
        )
        def _get_tools():
            try:
                loop = asyncio.get_running_loop()
                return loop.run_until_complete(load_mcp_tools(None, connection=connection))
            except RuntimeError:
                return asyncio.run(load_mcp_tools(None, connection=connection))
        _amap_tools = _get_tools()

        print(f"✅ 高德地图工具加载成功")
        print(f"   工具数量: {len(_amap_tools)}")

        # 打印可用工具列表
        if _amap_tools:
            print("   可用工具:")
            for tool in _amap_tools[:10]:
                print(f"     - {tool.name}")
            if len(_amap_tools) > 10:
                print(f"     ... 还有 {len(_amap_tools) - 10} 个工具")

    return _amap_tools


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """
    根据名称获取特定工具（支持模糊匹配）

    Args:
        name: 工具名称（可以是完整名称或部分名称）

    Returns:
        匹配的 BaseTool 或 None
    """
    tools = get_amap_tools()

    # 精确匹配
    for tool in tools:
        if tool.name == name:
            return tool

    # 模糊匹配：检查名称是否包含关键词
    for tool in tools:
        if name.lower() in tool.name.lower():
            return tool

    return None


def get_all_tool_names() -> List[str]:
    """获取所有可用工具的名称"""
    return [tool.name for tool in get_amap_tools()]


class AmapService:
    """高德地图服务封装类 - LangChain 版本"""

    def __init__(self):
        """初始化服务"""
        self.tools = get_amap_tools()

    def _invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用工具的辅助方法

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        tool = get_tool_by_name(tool_name)
        if not tool:
            raise ValueError(f"未找到工具: {tool_name}, 可用工具: {get_all_tool_names()}")

        try:
            result = tool.invoke(arguments)
            return result
        except Exception as e:
            raise RuntimeError(f"工具 {tool_name} 执行失败: {str(e)}")

    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """
        搜索POI景点

        Args:
            keywords: 搜索关键词
            city: 城市
            citylimit: 是否限制在城市范围内

        Returns:
            POI信息列表
        """
        try:
            result = self._invoke_tool(
                "maps_text_search",
                {
                    "keywords": keywords,
                    "city": city,
                    "citylimit": str(citylimit).lower()
                }
            )

            print(f"POI搜索结果: {str(result)[:200]}...")

            # TODO: 解析实际的POI数据
            return []

        except Exception as e:
            print(f"❌ POI搜索失败: {str(e)}")
            return []

    def get_weather(self, city: str) -> List[WeatherInfo]:
        """
        查询天气

        Args:
            city: 城市名称

        Returns:
            天气信息列表
        """
        try:
            result = self._invoke_tool(
                "maps_weather",
                {"city": city}
            )

            print(f"天气查询结果: {str(result)[:200]}...")

            # TODO: 解析实际的天气数据
            return []

        except Exception as e:
            print(f"❌ 天气查询失败: {str(e)}")
            return []

    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking"
    ) -> Dict[str, Any]:
        """
        规划路线

        Args:
            origin_address: 起点地址
            destination_address: 终点地址
            origin_city: 起点城市
            destination_city: 终点城市
            route_type: 路线类型 (walking/driving/transit)

        Returns:
            路线信息
        """
        try:
            # 根据路线类型选择工具
            tool_map = {
                "walking": "maps_direction_walking_by_address",
                "driving": "maps_direction_driving_by_address",
                "transit": "maps_direction_transit_integrated_by_address"
            }

            tool_name = tool_map.get(route_type, "maps_direction_walking_by_address")

            # 构建参数
            arguments = {
                "origin_address": origin_address,
                "destination_address": destination_address
            }

            # 公共交通需要城市参数
            if route_type == "transit":
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            else:
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city

            result = self._invoke_tool(tool_name, arguments)

            print(f"路线规划结果: {str(result)[:200]}...")

            # TODO: 解析实际的路线数据
            return {"raw": str(result)}

        except Exception as e:
            print(f"❌ 路线规划失败: {str(e)}")
            return {}

    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """
        地理编码（地址转坐标）

        Args:
            address: 地址
            city: 城市

        Returns:
            经纬度坐标
        """
        try:
            arguments = {"address": address}
            if city:
                arguments["city"] = city

            result = self._invoke_tool("maps_geo", arguments)

            print(f"地理编码结果: {str(result)[:200]}...")

            # TODO: 解析实际的坐标数据
            return None

        except Exception as e:
            print(f"❌ 地理编码失败: {str(e)}")
            return None

    def reverse_geocode(self, longitude: float, latitude: float) -> Optional[str]:
        """
        逆地理编码（坐标转地址）

        Args:
            longitude: 经度
            latitude: 纬度

        Returns:
            地址字符串
        """
        try:
            result = self._invoke_tool(
                "maps_regeo",
                {"longitude": longitude, "latitude": latitude}
            )

            print(f"逆地理编码结果: {str(result)[:200]}...")

            # TODO: 解析地址数据
            return str(result)

        except Exception as e:
            print(f"❌ 逆地理编码失败: {str(e)}")
            return None

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """
        获取POI详情

        Args:
            poi_id: POI ID

        Returns:
            POI详情信息
        """
        try:
            result = self._invoke_tool(
                "maps_search_detail",
                {"id": poi_id}
            )

            print(f"POI详情结果: {str(result)[:200]}...")

            # 尝试解析 JSON
            import json
            import re

            result_str = str(result)
            json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data

            return {"raw": result_str}

        except Exception as e:
            print(f"❌ 获取POI详情失败: {str(e)}")
            return {}

    def get_around(self, longitude: float, latitude: float, keywords: str = "") -> List[POIInfo]:
        """
        周边搜索

        Args:
            longitude: 中心点经度
            latitude: 中心点纬度
            keywords: 搜索关键词（可选）

        Returns:
            周边POI列表
        """
        try:
            arguments = {
                "longitude": longitude,
                "latitude": latitude
            }
            if keywords:
                arguments["keywords"] = keywords

            result = self._invoke_tool("maps_around", arguments)

            print(f"周边搜索结果: {str(result)[:200]}...")

            # TODO: 解析POI数据
            return []

        except Exception as e:
            print(f"❌ 周边搜索失败: {str(e)}")
            return []


# 创建全局服务实例
_amap_service: Optional[AmapService] = None


def get_amap_service() -> AmapService:
    """获取高德地图服务实例(单例模式)"""
    global _amap_service

    if _amap_service is None:
        _amap_service = AmapService()

    return _amap_service


def reset_amap_service():
    """重置服务实例（用于重新加载工具）"""
    global _amap_tools, _amap_service
    _amap_tools = None
    _amap_service = None
