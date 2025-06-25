from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

ResearchTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a research person helps me to find information about any topic. word limit = 150"),
        ("human", "Research about the topic {topic}")
    ]
)

# Things to perform

# Summarizer Chain
# → Returns a short summary of the input

# Sentiment Analysis Chain
# → Returns sentiment: Positive / Negative / Neutral

# Keyword Extraction Chain
# → Returns top 5 keywords from the text

def Summarizer(text):
    template = ChatPromptTemplate.from_template("Summarize the text in 50 words: {text}")
    return template.format_prompt(text=text)

def SentimentAnalysis(text):
    template = ChatPromptTemplate.from_template("What is the sentiment of the text in 1 line: {text}")
    return template.format_prompt(text=text)

def KeywordExtraction(text):
    template = ChatPromptTemplate.from_template("Extract top 5 keywords from the text in a list format: {text}")
    return template.format_prompt(text=text)

def create_blog(summary, sentiment, keywords):
    template = ChatPromptTemplate.from_template(
        "Stack all these with proper formatting: {summary}, sentiment: {sentiment}, and keywords: {keywords}"
    )
    prompt = template.format_prompt(summary=summary, sentiment=sentiment, keywords=keywords)
    return llm.invoke(prompt)

   
# These all are to take to text from the main chain and get the result out of it
SummarizerChain = RunnableLambda(lambda x: Summarizer(x)) | llm | StrOutputParser()
SentimentAnalysisChain = RunnableLambda(lambda x: SentimentAnalysis(x)) | llm | StrOutputParser()
KeywordExtractionChain = RunnableLambda(lambda x: KeywordExtraction(x)) | llm | StrOutputParser()

MainChain = (
    ResearchTemplate
    | llm
    | StrOutputParser()
    | RunnableParallel(branches={"summary" : SummarizerChain, "sentiment" : SentimentAnalysisChain, "keywords" : KeywordExtractionChain})
    | RunnableLambda(lambda x: create_blog(x["branches"]["summary"], x["branches"]["sentiment"], x["branches"]["keywords"]))     
    
)

result = MainChain.invoke({"topic": "Artificial Intelligence in Healthcare"})

print(result)