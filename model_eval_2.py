import mlflow
import pandas as pd
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Create singleton instances for models
llm_instance = None
judge_model_instance = None

def get_llm():
    global llm_instance
    if llm_instance is None:
        llm_instance = CTransformers(
            model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            model_type="mistral",
            config={'max_new_tokens': 128, 'temperature': 0.7}
        )
    return llm_instance

def get_judge_model():
    global judge_model_instance
    if judge_model_instance is None:
        judge_model_instance = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGUF",
            model_file="llama-2-7b-chat.Q4_K_M.gguf",
            model_type="llama",
            config={'max_new_tokens': 128, 'temperature': 0.7}
        )
    return judge_model_instance

# Define pre-canned metrics
def get_pre_canned_metrics():
    return [
        mlflow.metrics.genai.answer_similarity(),
        mlflow.metrics.genai.answer_correctness(),
        mlflow.metrics.genai.answer_relevance(),
        mlflow.metrics.genai.relevance(),
        mlflow.metrics.genai.faithfulness()
    ]

# Function to evaluate responses
def evaluate_response(user_input, response):
    judge_model = get_judge_model()

    # Prepare evaluation data
    data = {
        'input': [user_input],
        'output': [response],
        'ground_truth': [user_input]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Evaluate response
    evaluation_results = mlflow.evaluate(
        model=judge_model,
        data=df,
        extra_metrics=get_pre_canned_metrics()
    )
    
    return evaluation_results

# Function to append data to an Excel file
def append_to_excel(data, file_name="chatbot_responses_mlflow.xlsx"):
    if os.path.exists(file_name):
        existing_data = pd.read_excel(file_name)
        new_data = pd.concat([existing_data, data], ignore_index=True)
        new_data.to_excel(file_name, index=False)
    else:
        data.to_excel(file_name, index=False)

# Function to evaluate responses and log results to MLflow
def evaluate_and_log_to_mlflow(user_input, response):
    evaluation_results = evaluate_response(user_input, response)
    
    metrics = {
        "answer_similarity": evaluation_results.get('answer_similarity', 0),
        "answer_correctness": evaluation_results.get('answer_correctness', 0),
        "answer_relevance": evaluation_results.get('answer_relevance', 0),
        "relevance": evaluation_results.get('relevance', 0),
        "faithfulness": evaluation_results.get('faithfulness', 0)
    }
    
    with mlflow.start_run():
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    data = {
        "User Input": [user_input],
        "Response": [response],
        "Answer Similarity": [metrics["answer_similarity"]],
        "Answer Correctness": [metrics["answer_correctness"]],
        "Answer Relevance": [metrics["answer_relevance"]],
        "Relevance": [metrics["relevance"]],
        "Faithfulness": [metrics["faithfulness"]]
    }
    df = pd.DataFrame(data)
    append_to_excel(df)

# Function to generate responses
def generate_response(user_input):
    llm = get_llm()
    
    template = """[INST] <<SYS>>You're a friendly, empathetic virtual counselor. Match the user's tone and respond naturally, as if continuing an ongoing conversation with a friend. Keep responses brief (1-2 sentences). Ask thoughtful questions based on what the user says. Avoid formulaic greetings or responses. Be supportive and encourage the user to open up, but don't suggest specific tasks or relaxation techniques. Provide emotional and mental support.<</SYS>>{text}[/INST]"""
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    response = llm_chain.run(user_input)
    
    evaluate_and_log_to_mlflow(user_input, response)
    
    return response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = generate_response(user_input)
        print("\nChatbot Response:", response)
