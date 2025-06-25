from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

messages = [
    ("system", "First generate a number in the format \"number\":\"12\" "),
    ("human", "Generate a random number in the above format.")
]

template = ChatPromptTemplate.from_template("Tell me the square root of {number}.")

chain = llm | template | llm | StrOutputParser()

values = chain.invoke(messages)

print(values)


