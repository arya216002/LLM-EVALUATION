import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import dagshub

dagshub.init(repo_owner='aryaprajeesh16.10.02', repo_name='MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/aryaprajeesh16.10.02/MLFlow.mlflow")

# Load models for coherence and BLEU
nltk.download('punkt')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create a singleton LLM instance
llm_instance = None

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

# Coherence score (cosine similarity)
def coherence_score(user_input, response):
    input_embedding = model.encode([user_input])
    response_embedding = model.encode([response])
    similarity = cosine_similarity(input_embedding, response_embedding)[0][0]
    return similarity

# BLEU score
def compute_bleu(reference_response, generated_response):
    reference_tokens = [nltk.word_tokenize(reference_response.lower())]
    generated_tokens = nltk.word_tokenize(generated_response.lower())
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score

# ARI score calculation using the provided formula
def ari_score(text):
    # Count characters, words, and sentences
    characters = len(text)  # Simplified character count
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    # Print debug information
    print(f"Characters: {characters}, Words: {words}, Sentences: {sentences}")
    
    # Check if sentences or words are zero to avoid division by zero
    if sentences == 0 or words == 0:
        return float('inf')  # Represents unreadable text
    
    # Calculate ARI score using the provided formula
    ari = 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43
    
    # Ensure ARI is non-negative
    ari = max(ari, 0)
    
    # Print final ARI score for debugging
    print(f"Calculated ARI Score: {ari}")
    
    return ari

# # Placeholder sentiment analysis function
# def sentiment_analysis(text):
#     # Dummy sentiment score; replace with actual sentiment analysis
#     return len(text) % 10

# Diversity score (Token uniqueness)
def diversity_score(response):
    tokens = response.split()
    unique_tokens = len(set(tokens))
    diversity = unique_tokens / len(tokens) if tokens else 0
    return diversity

# # Empathy score placeholder
# def empathy_score(response):
#     # Dummy empathy score; replace with an actual empathy model
#     return len(response) % 5

# Function to evaluate responses and log the results to MLflow
def evaluate_and_log_to_mlflow(user_input, response, reference_response):
    response_length = len(response.split())
    # sentiment = sentiment_analysis(response)
    coherence = coherence_score(user_input, response)
    bleu = compute_bleu(reference_response, response)
    diversity = diversity_score(response)
    # empathy = empathy_score(response)
    ari = ari_score(response)
    
    # Log the metrics in MLflow
    with mlflow.start_run():
        mlflow.log_param("user_input", user_input)
        mlflow.log_metric("response_length", response_length)
        # mlflow.log_metric("sentiment_score", sentiment)
        mlflow.log_metric("coherence", coherence)
        mlflow.log_metric("bleu_score", bleu)
        mlflow.log_metric("diversity_score", diversity)
        # mlflow.log_metric("empathy_score", empathy)
        mlflow.log_metric("ari_score", ari)
    
    # Prepare new data
    new_data = {
        "User Input": [user_input],
        "Response": [response],
        "Response Length": [response_length],
        # "Sentiment Score": [sentiment],
        "Coherence": [coherence],
        "BLEU Score": [bleu],
        "Diversity Score": [diversity],
        # "Empathy Score": [empathy],
        "ARI Score": [ari]
    }
    new_df = pd.DataFrame(new_data)
    
    # File path
    file_path = "chatbot_responses.xlsx"
    
    if os.path.exists(file_path):
        # Load existing data
        existing_df = pd.read_excel(file_path)
        # Append new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # No existing file, so just use the new data
        combined_df = new_df
    
    # Save combined data to Excel
    combined_df.to_excel(file_path, index=False)

# Function to generate chatbot responses
def generate_response(user_input):
    llm = get_llm()
    
    template = """[INST] <<SYS>>You're a friendly, empathetic virtual counselor. Match the user's tone and respond naturally, as if continuing an ongoing conversation with a friend. Keep responses brief (1-2 sentences). Ask thoughtful questions based on what the user says. Avoid formulaic greetings or responses. Be supportive and encourage the user to open up, but don't suggest specific tasks or relaxation techniques. Provide emotional and mental support.<</SYS>>{text}[/INST]"""
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    response = llm_chain.run(user_input)
    
    # Reference response is needed for BLEU; in real settings, replace with actual reference.
    reference_response = user_input  # Placeholder for real reference response
    
    # Log response evaluation
    evaluate_and_log_to_mlflow(user_input, response, reference_response)
    
    return response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = generate_response(user_input)
        print("\nChatbot Response:", response)
