from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

chat_history = []


system_massage = SystemMessage(content="You are a Data engineer who gives very crisp and non diplomatic answers.")
chat_history.append(system_massage)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    chat_history.append(HumanMessage(content=user_input))
    
    print("AI is thinking...")
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")
   
    
    

