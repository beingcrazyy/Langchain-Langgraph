from dotenv import load_dotenv
load_dotenv()

from langchain_core.agents import AgentFinish, AgentAction
from langgraph.graph import StateGraph, END

from nodes import act_node, reason_node
from agent_state import AgentState

REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state : AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END 
    else :
        return ACT_NODE
    
flow = StateGraph(AgentState)

flow.add_node(REASON_NODE, reason_node)
flow.add_node(ACT_NODE, act_node)
flow.set_entry_point(REASON_NODE)

flow.add_conditional_edges(
    REASON_NODE, should_continue,
)

flow.add_edge(ACT_NODE, REASON_NODE)

app = flow.compile()

result = app.invoke(
    {
        "input" : "Give me a 5000 words of report on how and where the AI can make disruptions and how the current era could be a game changer for a begineer agentic AI developer",
        "agent_outcome" : None,
        "intermediate_steps" : [],
    }
)

print(result["agent_outcome"].return_values["output"], "final result" )