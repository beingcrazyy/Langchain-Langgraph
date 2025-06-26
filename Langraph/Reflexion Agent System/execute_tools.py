import json
from typing import List,Dict,Any
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(max_results = 5)

def execute_tools(state : List[BaseMessage]) -> List[BaseMessage]:
    last_ai_massage : AIMessage = state[-1]

    if not hasattr(last_ai_massage, "tool_calls") or not last_ai_massage.tool_calls:
        return[]
    
    tool_massages = []

    for tool_call in last_ai_massage.tool_calls:
        if tool_call["name"] in ["Answer", "ReviserAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries",[])


            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result

            tool_massages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id = call_id
                )
            )
    
    return tool_massages