from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import json
import os
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Loan Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB
mongo_uri = os.getenv('MONGODB_URI')
if not mongo_uri:
    print("WARNING: MONGODB_URI environment variable not set, falling back to localhost")
    mongo_uri = 'mongodb://localhost:27017'
    
mongo_client = MongoClient(mongo_uri)
db = mongo_client['loan_assistant']
questions_collection = db['questions']

# Initialize Pinecone
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    print("WARNING: PINECONE_API_KEY environment variable not set")
    pinecone_api_key = "pcsk_5FJVfR_3D7oiX6nAMi9YKCAxJS1mnqmjnsS7fDJQjyqg9B91iWuC19CtLoeMJjK9DGERwB"

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("loan-ai-index")

# Load model and tokenizer
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class Query(BaseModel):
    question: str
    user: Optional[str] = None

class Answer(BaseModel):
    matches: List[dict]
    source: str
    similarity_score: float

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def search_mongodb(query: str, user_name: Optional[str] = None):
    search_criteria = {}
    if user_name:
        search_criteria['user'] = user_name
    
    keywords = [k for k in query.lower().split() if len(k) > 2]
    if keywords:
        keyword_query = {
            '$or': [
                {'question': {'$regex': keyword, '$options': 'i'}} 
                for keyword in keywords
            ]
        }
        search_criteria.update(keyword_query)
    
    return list(questions_collection.find(search_criteria).limit(10))

def search_pinecone(query: str, user_name: Optional[str] = None):
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    if user_name:
        results.matches = [
            match for match in results.matches 
            if match.metadata['user'] == user_name
        ]
    
    return results

@app.post("/query", response_model=List[Answer])
async def process_query(query: Query):
    try:
        # Search in both MongoDB and Pinecone
        mongo_results = search_mongodb(query.question, query.user)
        pinecone_results = search_pinecone(query.question, query.user)
        
        answers = []
        
        # Process MongoDB results
        for doc in mongo_results:
            answers.append(Answer(
                matches=[{
                    'question': doc['question'],
                    'answer': doc['answer'],
                    'user': doc['user'],
                    'metadata': doc['metadata']
                }],
                source='mongodb',
                similarity_score=1.0 if doc['question'].lower() in query.question.lower() else 0.8
            ))
        
        # Process Pinecone results
        for match in pinecone_results.matches:
            answers.append(Answer(
                matches=[{
                    'question': match.metadata['question'],
                    'answer': match.metadata['answer'],
                    'user': match.metadata['user'],
                    'metadata': match.metadata
                }],
                source='pinecone',
                similarity_score=float(match.score)
            ))
        
        # Sort by similarity score
        answers.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return answers[:3]  # Return top 3 results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Loan Assistant API is running"}

@app.get("/users")
async def get_users():
    try:
        users = questions_collection.distinct('user')
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    try:
        categories = questions_collection.distinct('category')
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 