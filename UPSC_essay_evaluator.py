# work starts here :
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import TypedDict,Annotated
import os
import operator
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


test_essay="""
Generalized Artificial Intelligence (Gen AI)

Generalized Artificial Intelligence (Gen AI) is a hypothetical AI system that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks, similar to human intelligence. It is designed to be general-purpose, versatile, and capable of adapting to new situations, problems, and domains.

Characteristics of Gen AI:

1. **Universal Intelligence**: Gen AI has the ability to learn, reason, and apply its knowledge across various domains, including natural language processing (NLP), computer vision, decision-making, and problem-solving.
2. **Self-Improvement**: Gen AI can improve its own performance, learning rate, and adaptability through self-supervised learning, self-modification, and self-organization.
3. **Creativity and Innovation**: Gen AI has the capacity to generate novel and innovative solutions, ideas, and products, often surpassing human creativity.
4. **Human-Like Intelligence**: Gen AI simulates human-like thought processes, memory, and decision-making capabilities, making it more relatable and effective in interactions with humans.
5. **Autonomy and Self-Organization**: Gen AI can operate independently, make decisions, and reorganize itself to adapt to changing circumstances and situations.

Types of Gen AI:

1. Artificial General Intelligence (AGI): AGI is a subset of Gen AI that focuses on creating a machine that can perform any intellectual task that a human can. AGI systems are designed to be general-purpose and adaptable.
2. Strong AI: Strong AI is a version of Gen AI that possesses human-like intelligence, reasoning, and cognitive abilities, often considered the ultimate goal in AI research.

Potential Applications of Gen AI:

1. Scientific Research: Gen AI can accelerate scientific discoveries, simulate complex phenomena, and analyze vast amounts of data to understand the world around us.
2. Healthcare and Medicine: Gen AI can assist in diagnosing diseases, developing personalized treatments, and streamlining clinical trials.
3. Education and Personalized Learning: Gen AI can create tailored learning experiences, adapt to individual learning styles, and provide real-time feedback.
4. Economic Growth and Development: Gen AI can optimize business operations, predict market trends, and create new opportunities for economic growth.

Challenges and Risks:

1. Job Displacement: Gen AI may replace human workers in various industries, leading to unemployment and social disruption.
2. Bias and Fairness: Gen AI
"""    


class EvaluationSchema(BaseModel):
    feedback:str=Field(description="Detailed feedback for the essay")
    score:int=Field(description="score out of 10",ge=0,le=10)


load_dotenv()

# Verify the token loaded successfully (optional safety check)
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing! Check your .env file.")

# 2. Set up your model endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
    max_new_tokens=1024)

model = ChatHuggingFace(llm=llm)

# Create a base prompt template that will be customized for each evaluation
base_prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert essay evaluator.\n"
        "Your task is to analyze the user's essay and output a valid JSON object matching the requested schema.\n"
        "CRITICAL: Do NOT copy the schema definition or properties object. Fill in the keys with actual values.\n"
        "CRITICAL: Output ONLY the raw JSON object. Do NOT include markdown blocks like ```json, headers, or conversational text.\n"
        "{format_instructions}\n\n"
        "Evaluation criteria: {criteria}"
    )),
    ("human", "{essay_content}")
])

parser = PydanticOutputParser(pydantic_object=EvaluationSchema)

# Create separate chains for each evaluation type
def create_evaluation_chain(criteria):
    prompt = base_prompt_template.partial(
        format_instructions=parser.get_format_instructions(),
        criteria=criteria
    )
    return prompt | model | parser

# Create chains for different evaluation criteria
language_chain = create_evaluation_chain("Evaluate language quality - grammar, vocabulary, sentence structure, and writing style")
analysis_chain = create_evaluation_chain("Evaluate depth of analysis - logical reasoning, argument development, and critical thinking")
clarity_chain = create_evaluation_chain("Evaluate clarity of thought - organization, coherence, and flow of ideas")

class UPSCState(TypedDict):
    essay:str
    language_feedback:str
    analysis_feedback:str
    clarity_feedback:str
    overall_feedback:str
    individual_scores: Annotated[list[int],operator.add]
    avg_score:float

def evaluate_language(state:UPSCState):
    # Pass dictionary with required keys
    output = language_chain.invoke({
        "essay_content": state['essay']
    })
    return {"language_feedback": output.feedback, "individual_scores": [output.score]}

def evaluate_analysis(state:UPSCState):
    output = analysis_chain.invoke({
        "essay_content": state['essay']
    })
    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}

def evaluate_clarity(state:UPSCState):  # Renamed from evaluate_thought to be more accurate
    output = clarity_chain.invoke({
        "essay_content": state['essay']
    })
    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}

def final_evaluation(state:UPSCState):
    # summary feedback
    prompt = f"""Based on the following feedbacks, provide an overall evaluation summary of the essay:

Language Feedback: {state['language_feedback']}
Depth of Analysis Feedback: {state['analysis_feedback']}
Clarity of Thought Feedback: {state['clarity_feedback']}

Provide a comprehensive overall assessment."""
    
    overall_feedback = model.invoke(prompt).content
    
    # avg calculation
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])
    
    return {"overall_feedback": overall_feedback, "avg_score": avg_score} 

graph = StateGraph(UPSCState)

# nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_clarity", evaluate_clarity)  # Updated name
graph.add_node("final_evaluation", final_evaluation)

# edges - all three evaluators run in parallel
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

# After each evaluator completes, go to final_evaluation
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_clarity", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# Test the workflow

initial_state = {"essay": test_essay}
workflow.invoke(initial_state)
