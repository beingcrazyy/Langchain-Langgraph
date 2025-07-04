from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_react_agent, tool
from langchain_tavily import TavilySearch
import datetime
from langchain import hub

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

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

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools=tools, llm = llm, prompt= react_prompt)