from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

# to show the previous conversations
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

thread_id = "1"
# to show the current messages
user_input = st.chat_input("Type here" )


if user_input:
    # first add the message history
    st.session_state["message_history"].append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.text(user_input)


    
    config = {"configurable":{"thread_id":thread_id}}
    response = chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=config)
    ai_message = response["messages"][-1].content
    st.session_state["message_history"].append({"role":"assistant","content":ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)        
       


