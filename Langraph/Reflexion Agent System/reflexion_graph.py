from typing import List
from langchain_core.messages import BaseMessage,ToolMessage,HumanMessage
from langgraph.graph import END, MessageGraph


from chains import reviser_chain, responder_chain
from execute_tools import execute_tools

graph = MessageGraph()

MAX_ITERATIONS = 6

graph.add_node("Responder", responder_chain)
graph.add_node("ExecuteTools", execute_tools)
graph.add_node("Reviser",reviser_chain)

graph.add_edge("Responder","ExecuteTools")
graph.add_edge("ExecuteTools","Reviser")

def event_loop(state:List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits

    if num_iterations > MAX_ITERATIONS:
        return END  
    return "execute_tools"

graph.add_conditional_edges("Reviser", event_loop)

graph.set_entry_point("Responder")

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(
   "Write me a blog post on the impact of war situation going on between Israel and Iran on INDIA")

print(response[-1].tool_calls[0]["args"]["answer"])
print(response)