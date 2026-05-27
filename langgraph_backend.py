# very simple chatbot with the memory persistance
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API")


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages] # add messages is a reducer function which adds the messages in the state at the time of invokation.


def chat_node(state:ChatState):
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=groq_api_key,max_tokens=500)
    # take the user query from state
    messages=state["messages"]

    # send to llm
    response = model.invoke(messages)
    # response stored to the llm
    return {"messages":[response]}
# langgraph orchestration
checkPointer = MemorySaver() # to save the every messsages at the time of invokation. because every invokation uses the new messages states and starts from scratch

graph = StateGraph(ChatState)

# add node 
graph.add_node("chat_node",chat_node)

# add edges 
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot = graph.compile(checkpointer=checkPointer)



