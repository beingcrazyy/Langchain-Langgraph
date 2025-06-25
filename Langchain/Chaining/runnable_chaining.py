from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

messages = [
    ("system", "First generate a number in the format \"number\":\"12\" "),
    ("human", "Generate a random number greater then 100 in the above format. also the number should have a perfect square root.")
]

template = ChatPromptTemplate.from_template("Tell me the square root of {number}.")



get_number = RunnableLambda(lambda x: llm.invoke(x))
get_prompt = RunnableLambda(lambda x: template.invoke(x))
get_sq_root = RunnableLambda(lambda x: llm.invoke(x))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=get_number, middle=[get_prompt, get_sq_root], last=parse_output)

values = chain.invoke(messages)

print(values)


