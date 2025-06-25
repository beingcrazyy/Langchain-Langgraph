from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

messages = [
    ("system", "First generate some numbers in the format Ex : [\"number\":\"12\", \"number\":\"34\"] basically i just need a list of numbers with displaying the number in a key called 'number' in double quotes. no other text. dont give brackets for every number i need a collective bracket it should look like a list. no curly brackets. also the numbers should have a perfect square root."),
    ("human", "Generate 5 random numbers in the above format.")
]


result = llm.invoke(messages)

values = result.content

print(values)

template = "Tell me the square root of {number}."

prompt_template = ChatPromptTemplate.from_template(template)


for value in values.split(","):
    prompt = prompt_template.invoke({
        value
    })

    answer = llm.invoke(prompt)
    print(answer.content)
    






