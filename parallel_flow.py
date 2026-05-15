# parallel workflow

#Non LLM Workflow

from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class BatsmanState(TypedDict):
    #inputs
    runs: int
    balls:int
    fours:int
    sixes: int 

    sr: float # strike rate
    bpb: float # balls per boundary
    boundary_percent: float 
    summary:str

def calculate_sr(state:BatsmanState):
    sr=(state["runs"]/state["balls"])*100
    
    return {"sr":sr} # while handling the parallel nodes working never return entire state it gets parallel update issue.

def cal_bpb(state:BatsmanState):

    bpb=state["balls"]/(state["fours"]+state["sixes"])
    
    return {"bpb":bpb}
def cal_boundary_percent(state:BatsmanState):
    bp = (((state["fours"]*4)+(state["sixes"]*6))/state["runs"])*100
    
    return {"boundary_percent":bp}

def summary(state:BatsmanState):
    summary = f"""
                Strike rate = {state["sr"]}
                Balls per boundary = {state["bpb"]}
                Boundary percent = {state["boundary_percent"]}                
"""
    
    return {"summary":summary}

graph = StateGraph(BatsmanState)

# nodes
graph.add_node("cal_sr",calculate_sr)
graph.add_node("cal_bpb",cal_bpb)
graph.add_node("cal_bp",cal_boundary_percent)
graph.add_node("summary",summary)

# edges
graph.add_edge(START,"cal_sr")
graph.add_edge(START,"cal_bpb")
graph.add_edge(START,"cal_bp")

graph.add_edge("cal_sr","summary")
graph.add_edge("cal_bpb","summary")
graph.add_edge("cal_bp","summary")

graph.add_edge("summary",END)

app = graph.compile()
initial_state={
    "runs":100,
    "balls":50,
    "fours":6,
    "sixes":4
}

print(app.invoke(initial_state))

app
