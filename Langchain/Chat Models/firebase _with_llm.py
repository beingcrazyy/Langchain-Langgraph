from langchain_openai import ChatOpenAI
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

PROJECT_ID = "langchain-39526"
COLLECTION_NAME = "chat_history" 
SESSION_ID = "session123456" 

client = firestore.Client(project=PROJECT_ID)

chat_history = FirestoreChatMessageHistory(
    client=client,
    collection=COLLECTION_NAME,
    session_id=SESSION_ID,
)


print("Start chatting with the AI. Type 'exit' to quit.")

system_massage = SystemMessage(content="You are a Data engineer who gives very crisp and non diplomatic answers.")

chat_history.add_message(system_massage)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    chat_history.add_user_message(user_input)
    
    print("AI is thinking...")
    result = llm.invoke(chat_history.messages)
    response = result.content
    chat_history.add_ai_message(response)
    print(f"AI: {response}")
   
    
    

