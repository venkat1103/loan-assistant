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

def search_mongodb(query, user_name=None):
    """Search in MongoDB using text-based matching"""
    search_criteria = {}
    
    # Add user filter if specified
    if user_name:
        search_criteria['user'] = user_name
    
    # Add keyword search if there are keywords
    keywords = [k for k in query.lower().split() if len(k) > 2]  # Only use keywords longer than 2 chars
    if keywords:
        keyword_query = {
            '$or': [
                {'question': {'$regex': keyword, '$options': 'i'}} for keyword in keywords
            ]
        }
        search_criteria.update(keyword_query)
    
    # If no search criteria, return all documents (limited to 10)
    return list(questions_collection.find(search_criteria).limit(10))

def search_similar_questions(query, user_name=None, top_k=3):
    """Search using Pinecone's semantic search"""
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
    """Detect user mentioned in the query"""
    users = ['Venkat', 'Shay', 'Karthik']
    for user in users:
        if user.lower() in query.lower():
            return user
    return None

def combine_search_results(mongo_results, pinecone_results, query):
    """Combine and deduplicate results from both sources"""
    combined_results = []
    seen_questions = set()
    
    # Add Pinecone results first (they're semantically matched)
    for match in pinecone_results.matches:
        question = match.metadata['question']
        if question not in seen_questions:
            combined_results.append({
                'source': 'pinecone',
                'similarity': match.score,
                'question': question,
                'answer': match.metadata['answer'],
                'user': match.metadata['user'],
                'metadata': match.metadata
            })
            seen_questions.add(question)
    
    # Add MongoDB results
    for doc in mongo_results:
        if doc['question'] not in seen_questions:
            combined_results.append({
                'source': 'mongodb',
                'similarity': 1.0 if doc['question'].lower() in query.lower() else 0.8,
                'question': doc['question'],
                'answer': doc['answer'],
                'user': doc['user'],
                'metadata': doc['metadata']
            })
            seen_questions.add(doc['question'])
    
    # Sort by similarity score
    return sorted(combined_results, key=lambda x: x['similarity'], reverse=True)

def main():
    print("Welcome to the Enhanced Loan Assistant! Type 'exit' to quit.")
    print("You can ask questions about loans, EMIs, or any loan-related queries.")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'exit':
            break
            
        # Try to detect user from query
        user = get_user_from_query(query)
        
        print("\nSearching for relevant information...")
        
        # Search in both MongoDB and Pinecone
        mongo_results = search_mongodb(query, user)
        pinecone_results = search_similar_questions(query, user)
        
        # Combine results
        combined_results = combine_search_results(mongo_results, pinecone_results, query)
        
        if not combined_results:
            print("\nNo relevant information found. Please try rephrasing your question.")
            continue
            
        print("\nRelevant information found:")
        for i, result in enumerate(combined_results[:3], 1):  # Show top 3 results
            print(f"\n{i}. Match Score: {result['similarity']:.2f} (via {result['source']})")
            print(f"   Question: {result['question']}")
            print(f"   Answer: {result['answer']}")
            print(f"   User: {result['user']}")
            
            # Show additional metadata if available
            metadata = result['metadata']
            if 'loan_amount' in metadata:
                print(f"   Loan Amount: {metadata['loan_amount']}")
            if 'effective_interest' in metadata:
                print(f"   Interest Rate: {metadata['effective_interest']}")
            if 'emi_amount' in metadata:
                print(f"   EMI Amount: {metadata['emi_amount']}")
            if 'due_date' in metadata:
                print(f"   Due Date: {metadata['due_date']}")

if __name__ == "__main__":
    main() 