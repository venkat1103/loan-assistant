import json
from pymongo import MongoClient
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import os

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize MongoDB
mongo_client = MongoClient(config['mongodb_uri'])
db = mongo_client['loan_assistant']
questions_collection = db['questions']

# Initialize Pinecone
pc = Pinecone(api_key=config['pinecone_api_key'])
index = pc.Index("loan-ai-index")

# Load model and tokenizer
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def search_similar_questions(query, user_name=None, top_k=3):
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Filter results by user if specified
    if user_name:
        results.matches = [match for match in results.matches if match.metadata['user'] == user_name]
    
    return results

def get_user_from_query(query):
    # Simple user detection from query
    users = ['Venkat', 'Shay', 'Karthik']
    for user in users:
        if user.lower() in query.lower():
            return user
    return None

def main():
    print("Welcome to the Improved Loan Assistant! Type 'exit' to quit.")
    print("You can ask questions about loans, EMIs, or any loan-related queries.")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'exit':
            break
            
        # Try to detect user from query
        user = get_user_from_query(query)
        
        print("\nSearching for relevant information...")
        results = search_similar_questions(query, user)
        
        if not results.matches:
            print("\nNo relevant information found. Please try rephrasing your question.")
            continue
            
        print("\nRelevant information found:")
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. Similarity Score: {match.score:.2f}")
            print(f"   Question: {match.metadata['question']}")
            print(f"   Answer: {match.metadata['answer']}")
            print(f"   User: {match.metadata['user']}")
            
            # Show additional metadata if available
            if 'loan_amount' in match.metadata:
                print(f"   Loan Amount: {match.metadata['loan_amount']}")
            if 'effective_interest' in match.metadata:
                print(f"   Interest Rate: {match.metadata['effective_interest']}")

if __name__ == "__main__":
    main() 