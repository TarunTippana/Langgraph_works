# Sequential workflows 
from langgraph.graph import StateGraph,START,END
from typing import TypedDict


class BMIState(TypedDict):
    weight_kg: float
    height_m:float
    bmi:float
    bmi_type:str

def calculate_bmi(state: BMIState)-> BMIState:
    weight= state["weight_kg"]
    height= state["height_m"]
    bmi = weight/(height**2)

    state["bmi"] = round(bmi,2)
    return state

def bmi_type(state:BMIState) -> BMIState:
    bmi=state["bmi"]
    if bmi<25:
        state["bmi_type"] = "low"
        return state
    elif 25<bmi<34:
        state["bmi_type"]="normal"
        return state
    else:
        state["bmi_type"]="obese"
        return state  




graph = StateGraph(BMIState)
# add nodes to your graph
graph.add_node("cal_bmi",calculate_bmi)
graph.add_node("BMI_Type",bmi_type)
# add edges to your graph
graph.add_edge(START,"cal_bmi")
graph.add_edge("cal_bmi","BMI_Type")
graph.add_edge("BMI_Type",END)

# compile the graph
app = graph.compile()
wt=float(input("Enter the weight in kg"))
ht=float(input("Enter the height in m"))
result=app.invoke({'weight_kg':wt,"height_m":ht})
print(result)
app