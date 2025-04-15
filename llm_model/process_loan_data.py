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

def process_qa_data():
    # Load the QA dataset
    with open('loan_qa_dataset.json', 'r') as f:
        qa_data = json.load(f)
    
    # Process each category
    for category, questions in qa_data.items():
        for qa in questions:
            # Store in MongoDB
            mongo_doc = {
                'category': category,
                'user': qa['user'],
                'question': qa['question'],
                'answer': qa['answer'],
                'metadata': qa['metadata']
            }
            questions_collection.insert_one(mongo_doc)
            
            # Generate embedding and store in Pinecone
            embedding = get_embedding(qa['question'])
            pinecone_vector = {
                'id': f"{category}_{qa['user']}",
                'values': embedding,
                'metadata': {
                    'category': category,
                    'user': qa['user'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    **qa['metadata']
                }
            }
            index.upsert(vectors=[pinecone_vector])

def main():
    print("🚀 Starting data processing...")
    try:
        process_qa_data()
        print("✅ Successfully processed and stored data in MongoDB and Pinecone!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 