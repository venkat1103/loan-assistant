import os
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize Pinecone
pc = Pinecone(api_key=config['pinecone_api_key'])
index = pc.Index("loan-ai-index")

# Load the model and tokenizer
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def search_similar_questions(query, top_k=3):
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

def main():
    print("Welcome to the Loan Assistant! Type 'exit' to quit.")
    print("You can ask questions about loans, EMIs, or any loan-related queries.")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'exit':
            break
            
        print("\nSearching for relevant information...")
        results = search_similar_questions(query)
        
        print("\nRelevant information found:")
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. Similarity Score: {match.score:.2f}")
            print(f"   Question: {match.metadata['question']}")
            print(f"   Answer: {match.metadata['answer']}")
            print(f"   User: {match.metadata['user']}")

if __name__ == "__main__":
    main() 