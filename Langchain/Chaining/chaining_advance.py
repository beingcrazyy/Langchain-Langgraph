from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt1 = [
    ("system", "First generate a number in the format \"number\":\"12\" "),
    ("human", "Generate a random number greater then 100 in the above format. also the number should have a perfect square root.")
]

template = ChatPromptTemplate.from_template("Tell me the square root of {number}. just give me the square root value without any other text or explanation.")

multiplication_template = ChatPromptTemplate.from_template("Multiply the {Sq_root} by {value} ")

prepare_for_multiplication = RunnableLambda(lambda x: {"Sq_root" : x, "value" : x})

chain = llm | template | llm | StrOutputParser() | prepare_for_multiplication | multiplication_template | llm | StrOutputParser()

values = chain.invoke(prompt1)

print(values)