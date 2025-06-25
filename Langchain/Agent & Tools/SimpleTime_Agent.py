from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, tool
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime, timedelta

load_dotenv()

@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time in the specified format.
    """
    return datetime.now().strftime(format)

llm = ChatOpenAI(model="gpt-4o-mini")   
query = input("Enter your question: ")
PromptTemplate = hub.pull("hwchase17/react")

tools = [get_current_time]
agent = create_react_agent(llm, tools, PromptTemplate)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  
agent_executor.invoke({"input" : query})
