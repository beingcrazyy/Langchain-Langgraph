from typing import TypedDict, Annotated, List
from langchain_core.tools import tool
import datetime
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini")

search_tool = TavilySearch(search_depth = "basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

@tool
def Tavily_search_tool(query: str):
    """Search the web using Tavily. Use only and only if really needed"""
    return search_tool.run(query)

tools = [Tavily_search_tool, get_system_time]

agent_with_tools = llm.bind_tools(tools=tools)

class ToolChatState(TypedDict):
    messages : Annotated[List, add_messages]


def chatbot(state : ToolChatState):
    return{
        "messages" : [agent_with_tools.invoke(state["messages"])]
    }


def execute_tools(state: ToolChatState):
    last_message = state["messages"][-1]

    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return END
    
tool_node = ToolNode(tools=tools)


graph = StateGraph(ToolChatState)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", execute_tools)
graph.add_edge("tool_node", "chatbot")


app = graph.compile()

while True:
    user_input = input("user:")
    if (user_input in ["exit", "end", "bye"]):
        break
    else:
        result = app.invoke({
            "messages" : [HumanMessage(content = user_input)]
        })

        print(result["messages"][-1].content)


