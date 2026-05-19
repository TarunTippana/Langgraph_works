# streamlit_app.py
import streamlit as st
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API")

# Initialize model
model = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, max_tokens=1000)

# Define schemas
class SentimentSchema(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="sentiment of the review.")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

struct_model = model.with_structured_output(SentimentSchema)
struct_model2 = model.with_structured_output(DiagnosisSchema)

# Define state
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

# Define functions
def find_sentiment(state: ReviewState):
    prompt = f"You are a sentiment analyser and find the sentiment of the given review \n Review: {state['review']}"
    senti = struct_model.invoke(prompt).sentiment 
    return {"sentiment": senti.lower()}

def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"
    
def positive_response(state: ReviewState):
    prompt = f"""Write a warm thank you message in response to this review:\n\n {state['review']}\n\n
    Also kindly ask the user to leave feedback on our website."""
    
    response = model.invoke(prompt).content
    return {"response": response}

def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose the negative review\n\n{state['review']}\n 
                return issue_type, tone and urgency."""
    response = struct_model2.invoke(prompt)
    return {"diagnosis": response.model_dump()}

def negative_response(state: ReviewState):
    diagnosis = state['diagnosis']
    
    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content
    return {'response': response}

# Build graph
def build_graph():
    graph = StateGraph(ReviewState)
    
    graph.add_node('find_sentiment', find_sentiment)
    graph.add_node('positive_response', positive_response)
    graph.add_node('run_diagnosis', run_diagnosis)
    graph.add_node('negative_response', negative_response)
    
    graph.add_edge(START, 'find_sentiment')
    graph.add_conditional_edges('find_sentiment', check_sentiment)
    graph.add_edge('positive_response', END)
    graph.add_edge('run_diagnosis', 'negative_response')
    graph.add_edge('negative_response', END)
    
    return graph.compile()

# Streamlit UI
st.set_page_config(
    page_title="AI Review Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Review Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("This AI assistant analyzes customer reviews, detects sentiment, and generates appropriate responses.")
    
    st.subheader("📊 What it does:")
    st.markdown("""
    - ✅ **Positive Reviews**: Sends warm thank you messages
    - 🔍 **Negative Reviews**: Diagnoses issues (UX, Performance, Bug, Support, Other)
    - 🎭 **Tone Analysis**: Identifies user's emotional state
    - ⚡ **Urgency Detection**: Determines issue priority
    - 💬 **Smart Responses**: Generates empathetic replies
    """)
    st.subheader("Tech stack used")
    st.markdown("""
    -Core Technologies
        Python - Primary programming language

        Streamlit - Web application framework for building the UI

        LLM & AI Framework
        LangGraph - Workflow orchestration and state management

        LangChain Groq - LangChain integration with Groq's LLM API

        Groq (Llama 3.1 8B) - Large Language Model for sentiment analysis, diagnosis, and response generation

        Data Validation & Schema
        Pydantic - Data validation and schema definition using BaseModel and Field

        Environment & Configuration
        python-dotenv - Environment variable management (API keys)

        Type System
            typing (Literal, TypedDict) - Type hints and state definitions
        
        Architecture Pattern
            Graph-based workflow using LangGraph's StateGraph

            Conditional edge routing for sentiment-based decision making

            Structured output from LLM using Pydantic schemas""")
    
    if st.button("🔄 Clear Review", use_container_width=True):
        st.session_state.review_text = ""
        st.session_state.processed = False
        st.rerun()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Customer Review")
    review_input = st.text_area(
        "Enter or paste the customer review here:",
        height=200,
        placeholder="Example: The product is really bad, it keeps crashing and the support team never responds...",
        key="review_text"
    )
    
    process_button = st.button("🚀 Process Review", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Analysis Results")
    
    if process_button and review_input:
        with st.spinner("Analyzing review..."):
            try:
                # Build and run the workflow
                workflow = build_graph()
                initial_state = {"review": review_input}
                result = workflow.invoke(initial_state)
                
                # Store results in session state
                st.session_state.processed = True
                st.session_state.result = result
                st.session_state.sentiment = result.get('sentiment', 'unknown')
                st.session_state.diagnosis = result.get('diagnosis', {})
                st.session_state.response = result.get('response', '')
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.processed = False
    
    # Display results
    if st.session_state.get('processed', False):
        result = st.session_state.result
        
        # Sentiment Display
        sentiment = result.get('sentiment', 'unknown')
        if sentiment == 'positive':
            st.success("✅ **Sentiment: Positive**")
        else:
            st.error("⚠️ **Sentiment: Negative**")
        
        # Diagnosis for negative reviews
        if result.get('diagnosis'):
            st.markdown("---")
            st.subheader("🔍 Issue Diagnosis")
            
            diagnosis = result['diagnosis']
            
            # Create metrics for diagnosis
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Issue Type", diagnosis.get('issue_type', 'N/A'))
            with col_b:
                st.metric("Tone", diagnosis.get('tone', 'N/A'))
            with col_c:
                urgency_color = "🔴" if diagnosis.get('urgency') == 'high' else "🟡" if diagnosis.get('urgency') == 'medium' else "🟢"
                st.metric("Urgency", f"{urgency_color} {diagnosis.get('urgency', 'N/A').title()}")
        
        st.markdown("---")
        
        # Generated Response
        st.subheader("💬 AI Generated Response")
        response_text = result.get('response', 'No response generated')
        st.info(response_text)
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📋 Copy Response", use_container_width=True):
                st.write("Response copied to clipboard!")
                st.toast("Response copied!", icon="✅")
        with col_btn2:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.processed = False
                st.rerun()
                
    elif process_button and not review_input:
        st.warning("⚠️ Please enter a review to analyze.")
    elif not process_button and not st.session_state.get('processed', False):
        st.info("👈 Enter a review and click 'Process Review' to see the analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Powered by LangGraph & Groq LLM | AI Review Assistant v1.0
    </div>
    """,
    unsafe_allow_html=True
)