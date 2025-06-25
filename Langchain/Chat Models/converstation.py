from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="You are a Data engineer."),
    HumanMessage(content="List down all the major tasks for a data analyst."),
]

result = llm.invoke(messages)

print(result)