# Sequential workflows using non LLM method
from langgraph.graph import StateGraph,START,END
from typing import TypedDict


class BMIState(TypedDict):
    weight_kg: float
    height_m:float
    bmi:float

def calculate_bmi(state: BMIState)-> BMIState:
    weight= state["weight_kg"]
    height= state["height_m"]
    bmi = weight/(height**2)

    state["bmi"] = round(bmi,2)
    return state

graph = StateGraph(BMIState)
# add nodes to your graph
graph.add_node("cal_bmi",calculate_bmi)
# add edges to your graph
graph.add_edge(START,"cal_bmi")
graph.add_edge("cal_bmi",END)

# compile the graph
app = graph.compile()
wt=float(input("Enter the weight in kg"))
ht=float(input("Enter the height in m"))
result=app.invoke({'weight_kg':wt,"height_m":ht})
print(result)