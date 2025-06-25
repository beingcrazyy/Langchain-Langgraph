from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableBranch

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

feedback_classification_prompt = ChatPromptTemplate.from_template(
    "Classify the following feedback as positive, negative, neutral or escalate: {feedback}")



positive_feedback_template = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a customer support agent. Word limit :15 "),
    ("human", "The customer has given positive feedback: {feedback}. Please respond with a short thank you message."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a customer support agent.Word limit :15 "),    
    ("human", "The customer has given negative feedback: {feedback}. Please respond with an short apology and a solution."),      
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a customer support agent.Word limit :15 "),
    ("human", "The customer has given neutral feedback: {feedback}. Please respond with a short standard acknowledgment message."),
    ]
)   

escelate_feedback_template = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a customer support agent. Word limit :15 "),    
    ("human", "The customer has given feedback that needs to be escalated: {feedback}. Please respond with a short message indicating that the issue will be escalated to a supervisor."),     
    ]   
)

get_reply = RunnableBranch(
    (
        lambda x : "postive" in x,
        positive_feedback_template | llm | StrOutputParser()
    ),
    (
        lambda x : "negative" in x,
        negative_feedback_template | llm | StrOutputParser()
    ),
    (
        lambda x : "neutral" in x,
        neutral_feedback_template | llm | StrOutputParser()
    ),
    
    escelate_feedback_template | llm | StrOutputParser()
    
)

feedback = "I just hate your product, it is not working as expected"

feedback_type_chain = feedback_classification_prompt | llm | StrOutputParser()

reply = feedback_type_chain | get_reply 

reply = reply.invoke({"feedback": feedback})

print(reply)