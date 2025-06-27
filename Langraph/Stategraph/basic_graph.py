from typing import TypedDict
from langgraph.graph import END, StateGraph

class CountState(TypedDict):
    count : int

def increment(state: CountState) -> CountState:
    return{
        "count" : state["count"]+1
    }

def should_continue(state):
    if (state["count"] < 5):
        return "increment"
    else :
        return END

graph = StateGraph(CountState)

graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.add_conditional_edges("increment", should_continue )

app = graph.compile()

state = {
    "count" : 0
}

result = app.invoke(state)
print(result)

