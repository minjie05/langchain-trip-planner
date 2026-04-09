"""多智能体旅行规划系统 - LangChain 版本"""

import asyncio
import json
from typing import Dict, Any, List, Optional, TypedDict
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.sessions import StdioConnection
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from ..services.llm_service import get_llm
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel
from ..config import get_settings

# ============ Agent提示词（LangChain ReAct 格式） ============

ATTRACTION_AGENT_PROMPT = """你是一个专业的景点搜索助手。根据用户的需求，使用 maps_text_search 工具搜索合适的景点。

**重要提示:**
1. 必须使用工具来搜索景点，不要自己编造景点信息
2. 搜索时提供准确的关键词和城市名称
3. 尽量多搜索几个景点，返回丰富的结果供用户选择
"""

WEATHER_AGENT_PROMPT = """你是一个专业的天气查询助手。根据用户的需求，使用 maps_weather 工具查询指定城市的天气信息。

**重要提示:**
1. 必须使用工具来查询天气，不要自己编造天气信息
2. 查询准确的日期和城市
3. 返回完整的天气信息
"""

HOTEL_AGENT_PROMPT = """你是一个专业的酒店推荐助手。根据用户的需求和偏好，使用 maps_text_search 工具搜索合适的酒店。

**重要提示:**
1. 必须使用工具来搜索酒店，不要自己编造酒店信息
2. 搜索时使用"酒店"或"宾馆"作为关键词
3. 结合用户偏好（如经济型、豪华型等）进行搜索
"""

PLANNER_AGENT_PROMPT = """你是专业的行程规划专家。你的任务是根据景点信息、天气信息和酒店信息，生成详细的旅行计划。

**旅行计划要求:**
1. 每天安排2-3个景点，考虑景点之间的距离和游览时间
2. 每天必须包含早中晚三餐
3. 每天推荐一个具体的酒店
4. 考虑天气因素安排行程
5. 提供实用的旅行建议

**输出格式:**
请返回完整的旅行计划，包含以下信息：
- 城市名称
- 行程天数
- 每天的详细安排（景点、餐饮、住宿）
- 天气信息
- 预算估算
- 总体建议
"""


# ============ LangGraph 状态定义 ============

class TripPlanningState(TypedDict):
    """旅行规划状态"""
    city: str
    start_date: str
    end_date: str
    travel_days: int
    transportation: str
    accommodation: str
    preferences: List[str]
    free_text_input: Optional[str]

    attractions_result: Optional[str]
    weather_result: Optional[str]
    hotels_result: Optional[str]

    final_plan: Optional[str]


# ============ LangGraph 节点函数 ============

def create_attraction_node(llm, tools):
    """创建景点搜索节点"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的景点搜索助手。搜索景点时使用 maps_text_search 工具。

**重要提示:**
1. 必须使用工具来搜索景点，不要自己编造景点信息
2. 搜索时提供准确的关键词和城市名称
3. 尽量多搜索几个景点，返回丰富的结果供用户选择

你拥有以下工具:
{tools}
工具名称: {tool_names}"""),
        ("placeholder", "{messages}")
    ]).partial(tools=str([t.name for t in tools]), tool_names=", ".join([t.name for t in tools]))

    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

    def node(state: TripPlanningState) -> Dict:
        keywords = state.get("preferences", ["景点"])
        keyword = keywords[0] if keywords else "景点"
        city = state["city"]

        query = f"请搜索{city}的{keyword}相关景点，返回尽可能多的景点信息，包括名称、地址、描述等。"

        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        return {"attractions_result": result["messages"][-1].content}
    return node


def create_weather_node(llm, tools):
    """创建天气查询节点"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的天气查询助手。查询天气时使用 maps_weather 工具。

**重要提示:**
1. 必须使用工具来查询天气，不要自己编造天气信息
2. 查询准确的日期和城市
3. 返回完整的天气信息

你拥有以下工具:
{tools}
工具名称: {tool_names}"""),
        ("placeholder", "{messages}")
    ]).partial(tools=str([t.name for t in tools]), tool_names=", ".join([t.name for t in tools]))

    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

    def node(state: TripPlanningState) -> Dict:
        city = state["city"]
        query = f"请查询{city}的天气信息，返回完整的天气预报。"

        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        return {"weather_result": result["messages"][-1].content}
    return node


def create_hotel_node(llm, tools):
    """创建酒店推荐节点"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的酒店推荐助手。搜索酒店时使用 maps_text_search 工具。

**重要提示:**
1. 必须使用工具来搜索酒店，不要自己编造酒店信息
2. 搜索时使用"酒店"或"宾馆"作为关键词
3. 结合用户偏好（如经济型、豪华型等）进行搜索

你拥有以下工具:
{tools}
工具名称: {tool_names}"""),
        ("placeholder", "{messages}")
    ]).partial(tools=str([t.name for t in tools]), tool_names=", ".join([t.name for t in tools]))

    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

    def node(state: TripPlanningState) -> Dict:
        city = state["city"]
        accommodation = state.get("accommodation", "酒店")

        query = f"请搜索{city}的{accommodation}，返回尽可能多的酒店信息，包括名称、地址、价格范围、评分等。"

        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        return {"hotels_result": result["messages"][-1].content}
    return node


def create_planner_node(llm):
    """创建行程规划节点（无工具，纯 LLM）"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_AGENT_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])

    def node(state: TripPlanningState) -> Dict:
        query = f"""请根据以下信息生成{state['city']}的{state['travel_days']}天旅行计划：

**基本信息:**
- 城市: {state['city']}
- 日期: {state['start_date']} 至 {state['end_date']}
- 天数: {state['travel_days']}天
- 交通方式: {state['transportation']}
- 住宿: {state['accommodation']}
- 偏好: {', '.join(state['preferences']) if state['preferences'] else '无'}

**景点信息:**
{state.get('attractions_result', '无')}

**天气信息:**
{state.get('weather_result', '无')}

**酒店信息:**
{state.get('hotels_result', '无')}
"""

        if state.get("free_text_input"):
            query += f"\n**额外要求:** {state['free_text_input']}"

        response = llm.invoke(query)
        return {"final_plan": response.content}
    return node


# ============ 主类 ============

class MultiAgentTripPlanner:
    """多智能体旅行规划系统 - LangChain 版本"""

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化多智能体旅行规划系统 (LangChain)...")

        try:
            settings = get_settings()
            self.llm = get_llm()

            # 不在这里加载 tools，改为延迟到 async_initialize 中加载
            self.tools = None
            self._initialized = False

            # 预先准备好连接参数
            self._connection_params = {
                "command": "uvx",
                "args": ["amap-mcp-server"],
                "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key},
                "transport": "stdio"
            }

            print("✅ 多智能体系统实例已创建（tools 将在首次调用时异步加载）")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    async def async_initialize(self):
        """异步懒加载 tools 并初始化节点（仅调用一次）"""
        if self._initialized:
            return

        print("  - 加载高德地图 MCP 工具...")
        connection = StdioConnection(**self._connection_params)
        self.tools = await load_mcp_tools(None, connection=connection)

        print(f"   工具数量: {len(self.tools)}")
        for tool in self.tools[:5]:
            print(f"     - {tool.name}")

        # 创建各个节点的 Agent
        print("  - 创建景点搜索节点...")
        self.attraction_node = create_attraction_node(self.llm, self.tools)

        print("  - 创建天气查询节点...")
        self.weather_node = create_weather_node(self.llm, self.tools)

        print("  - 创建酒店推荐节点...")
        self.hotel_node = create_hotel_node(self.llm, self.tools)

        print("  - 创建行��规划节点...")
        self.planner_node = create_planner_node(self.llm)

        # 构建 LangGraph
        print("  - 构建工作流图...")
        self.graph = self._build_graph()

        self._initialized = True
        print(f"✅ 多智能体系统初始化完成")

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(TripPlanningState)

        # 添加节点
        workflow.add_node("attraction", self.attraction_node)
        workflow.add_node("weather", self.weather_node)
        workflow.add_node("hotel", self.hotel_node)
        workflow.add_node("planner", self.planner_node)

        # 定义并行执行：景点和天气可以同时查询，酒店也可以并行
        # 然后汇总到 planner
        workflow.add_edge("attraction", "planner")
        workflow.add_edge("weather", "planner")
        workflow.add_edge("hotel", "planner")
        workflow.add_edge("planner", END)

        # 设置入口
        workflow.set_entry_point("attraction")

        # 编译图
        checkpointer = MemorySaver()
        graph = workflow.compile(checkpointer=checkpointer)

        return graph

    async def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用 LangGraph 多智能体协作生成旅行计划

        Args:
            request: 旅行请求

        Returns:
            旅行计划
        """
        try:
            # 确保异步初始化已完成（懒加载）
            await self.async_initialize()

            print(f"\n{'='*60}")
            print(f"🚀 开始 LangGraph 多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"{'='*60}\n")

            # 准备初始状态
            initial_state: TripPlanningState = {
                "city": request.city,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "travel_days": request.travel_days,
                "transportation": request.transportation,
                "accommodation": request.accommodation,
                "preferences": request.preferences or [],
                "free_text_input": request.free_text_input,
                "attractions_result": None,
                "weather_result": None,
                "hotels_result": None,
                "final_plan": None
            }

            # 使用已初始化的节点执行搜索
            print("📍 查询景点...")
            attraction_state = initial_state.copy()
            attraction_result_dict = self.attraction_node(attraction_state)
            attractions_result = attraction_result_dict.get("attractions_result", "")

            print("📍 查询天气...")
            weather_state = initial_state.copy()
            weather_result_dict = self.weather_node(weather_state)
            weather_result = weather_result_dict.get("weather_result", "")

            print("📍 搜索酒店...")
            hotel_state = initial_state.copy()
            hotel_result_dict = self.hotel_node(hotel_state)
            hotels_result = hotel_result_dict.get("hotels_result", "")

            print(f"景点搜索完成: {attractions_result[:200]}...")
            print(f"天气查询完成: {weather_result[:200]}...")
            print(f"酒店搜索完成: {hotels_result[:200]}...\n")

            # 步骤4: 行程规划 Agent 整合信息生成计划
            print("📋 生成行程计划...")
            planner_query = self._build_planner_query(
                request, attractions_result, weather_result, hotels_result
            )
            planner_response = self.planner_node({
                **initial_state,
                "attractions_result": attractions_result,
                "weather_result": weather_result,
                "hotels_result": hotels_result
            })

            final_plan = planner_response.get("final_plan", "")
            print(f"行程规划完成: {final_plan[:300]}...\n")

            # 解析最终计划
            trip_plan = self._parse_response(final_plan, request)

            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")

            return trip_plan

        except Exception as e:
            print(f"❌ 生成旅行计划失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)

    def _build_planner_query(
        self,
        request: TripRequest,
        attractions: str,
        weather: str,
        hotels: str = ""
    ) -> str:
        """构建行程规划查询"""
        query = f"""请根据以��信息生成{request.city}的{request.travel_days}天旅行计划：

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点信息:**
{attractions}

**天气信息:**
{weather}

**酒店信息:**
{hotels}

**要求:**
1. 每天安排2-3个景点
2. 每天必须包含早中晚三餐
3. 每天推荐一个具体的酒店（从酒店信息中选择）
4. 考虑景点之间的距离和交通方式
5. 返回完整的旅行计划内容
6. 景点的经纬度坐标要真实准确
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query

    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """
        解析 Agent 响应

        Args:
            response: Agent 响应文本
            request: 原始请求

        Returns:
            旅行计划
        """
        try:
            # 尝试从响应中提取 JSON
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                # 如果找不到 JSON，返回备用计划
                print("⚠️  响应中未找到 JSON 数据，使用备用方案")
                return self._create_fallback_plan(request)

            # 解析 JSON
            data = json.loads(json_str)

            # 转换为 TripPlan 对象
            trip_plan = TripPlan(**data)

            return trip_plan

        except Exception as e:
            print(f"⚠️  解析响应失败: {str(e)}")
            print(f"   将使用备用方案生成计划")
            return self._create_fallback_plan(request)

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划（当 Agent 失败时）"""
        from datetime import datetime, timedelta

        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")

        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)

            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(
                            longitude=116.4 + i*0.01 + j*0.005,
                            latitude=39.9 + i*0.01 + j*0.005
                        ),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(
                        type="breakfast",
                        name=f"第{i+1}天早餐",
                        description="当地特色早餐",
                        estimated_cost=30
                    ),
                    Meal(
                        type="lunch",
                        name=f"第{i+1}天午餐",
                        description="午餐推荐",
                        estimated_cost=50
                    ),
                    Meal(
                        type="dinner",
                        name=f"第{i+1}天晚餐",
                        description="晚餐推荐",
                        estimated_cost=80
                    )
                ]
            )
            days.append(day_plan)

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程，建议提前查看各景点的开放时间。"
        )


# 全局多智能体系统实例
_multi_agent_planner: Optional[MultiAgentTripPlanner] = None


async def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例（单例模式，异步懒加载）"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()

    # 确保 tools 已异步加载
    await _multi_agent_planner.async_initialize()

    return _multi_agent_planner


async def reset_trip_planner_agent():
    """重置 Agent 实例（用于重新初始化）"""
    global _multi_agent_planner
    _multi_agent_planner = None
