from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import datetime
from schema import Answer,ReviserAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

pydantic_parser = PydanticToolsParser(tools=[Answer])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert AI researcher of {time}

            1. {first_instruction}
            2. Reflect & Critique your answer. Be severe to maximize improvement
            3. After the reflection, **List 1-3 search Queries seprately** for researching improvements. do not include them inside the reflection
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system", "Answer the user's question above using the required format"
        )
    ]
).partial(
    time= lambda : datetime.datetime.now().isoformat()
)

reviser_prompt = """
                Revise your previous answer using a nes information.
                You should use the pervious critique to add imported information to the answer
                You must include numerical citations in your revised answer to insure that it is verified.
                Add a "Sources" section at the bottom of your answer(which does not count the word limit). in form of 
                1. https//example.com
                2. https//example.com
                Yoy should use the previous critique to remove superfloues information from your answer and make sure it should be around 250 words only.
                """



responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed answer to the question"
)

reviser_prompt_template = actor_prompt_template.partial(
    first_instruction = reviser_prompt
)

llm = ChatOpenAI(model = "gpt-4o-mini")

responder_chain = responder_prompt_template | llm.bind_tools(tools = [Answer], tool_choice = "Answer") 

validation = PydanticToolsParser(tools = [Answer])

reviser_chain = reviser_prompt_template | llm.bind_tools(tools=[ReviserAnswer], tool_choice="ReviserAnswer")

# response = responder_chain.invoke({
#     "messsages" : [HumanMessage(content="Write me a blog post on The war situation going on between Israel and Iran")]
# })

# print(response)