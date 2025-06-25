from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from tavily import TavilyClient
import datetime
import os

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_current_time() -> str:
    """Get the current time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def tavily_search(query: str) -> str:
    """Use Tavily to search the web."""
    results = tavily_client.search(query=query)
    return results["results"][0]["content"] if results["results"] else "No results found."


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

tools = [tavily_search, get_current_time]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

result = agent.run("Which company made the first COVID-19 vaccine and when was it made? How many years ago was it?")
print(result)

# from langchain.agents import initialize_agent, Tool
# from langchain.agents.agent_types import AgentType
# from langchain.chat_models import ChatOpenAI
# from tavily import TavilyClient
# from dotenv import load_dotenv
# import datetime
# import os

# load_dotenv()

# # Setup LLM
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

# # Tavily Search Tool
# tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# def web_search(query: str) -> str:
#     results = tavily_client.search(query=query)
#     return results["results"][0]["content"] if results["results"] else "No results found."


# def get_current_time(_: str) -> str:
#     return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# tools = [
#     Tool(name="Web Search", func=web_search, description="Search the web for current information"),
#     Tool(name="Current Time", func=get_current_time, description="Get the current date and time")
# ]


# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )


# response = agent.run("Which country made the most effective COVID-19 vaccine and when was it made? How many years ago was it?")
# print(response)
