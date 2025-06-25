from typing_extensions import Annotated, TypedDict
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class Joke(TypedDict):
    """Joke to tell user"""

    opening: Annotated[str, ..., "The setup of the joke"]
    climax: Annotated[str, ..., "The punchline of the joke"]


messages = [
    ("system", "You are the most funny person around the globe and know 100s of crazyy jokes and you never repeat a joke"),
    ("human", "Tell me another joke")
]

# result_true = llm.invoke(messages)
structured_llm = llm.with_structured_output(Joke)
result = structured_llm.invoke(messages)
print(result)