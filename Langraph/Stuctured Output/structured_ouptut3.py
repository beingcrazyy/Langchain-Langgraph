json_schema = {
    "title" : "joke",
    "description" : "Format of the joke",
    "type" : "object",
    "properties" : {
        "setup" : {
            "type" : "string",
            "description" : "The Story of the joke"
        },
        "Climax" : {
            "type" : "string",
            "description" : "The punchline of the joke"       
         }
    },
    "required" : ["setup", "Climax"]
}

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

messages = [
    ("system", "You are the most funny person around the globe and know 100s of crazyy jokes and you never repeat a joke"),
    ("human", "Tell me another joke")
]

# result_true = llm.invoke(messages)
structured_llm = llm.with_structured_output(json_schema)
result = structured_llm.invoke(messages)

print(result)

