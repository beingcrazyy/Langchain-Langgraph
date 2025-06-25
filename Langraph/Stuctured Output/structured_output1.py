from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class Joke(BaseModel):
    """The way you can consider to tell a joke also this is just to tell one joke if asked multiple duplicate this format for multiple joke too"""

    opening : str = Field(description="Opening and the story of the joke")
    climax : str = Field (description="The part where the punch comes")


messages = [
    ("system", "You are the most funny person around the globe and know 100s of crazyy jokes and you never repeat a joke"),
    ("human", "Tell me a joke in hinglish")
]

# result_true = llm.invoke(messages)
structured_llm = llm.with_structured_output(Joke)
result = structured_llm.invoke(messages)

print(result)
# print(llm.invoke("Tell me 3 jokes"))
    