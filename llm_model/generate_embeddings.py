import os
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_5FJVfR_3D7oiX6nAMi9YKCAxJS1mnqmjnsS7fDJQjyqg9B91iWuC19CtLoeMJjK9DGERwB")

# Get the index
index = pc.Index("loan-ai-index")

# Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_data():
    """Load data from loan_dataset.jsonl"""
    data = []
    file_path = os.path.join(os.path.dirname(__file__), "loan_dataset.jsonl")
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def prepare_vectors(data):
    """Prepare vectors for embedding"""
    vectors = []
    for item in data:
        # Combine context and question for better semantic search
        text = f"{item['context']} {item['question']}"
        
        # Tokenize and generate embeddings
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Convert to list for Pinecone
        vector = sentence_embeddings[0].tolist()
        
        # Prepare metadata
        metadata = {
            'user': item['user'],
            'context': item['context'],
            'question': item['question'],
            'answer': item['answer']
        }
        
        vectors.append((vector, metadata))
    return vectors

def upload_to_pinecone(vectors, batch_size=100):
    """Upload vectors to Pinecone in batches"""
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        # Prepare batch for upsert
        upsert_data = []
        for j, (vector, metadata) in enumerate(batch):
            upsert_data.append({
                'id': f'vec_{i+j}',
                'values': vector,
                'metadata': metadata
            })
        # Upload batch
        index.upsert(vectors=upsert_data)
        print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")

def main():
    print("🚀 Starting data processing and embedding generation...")
    try:
        # Load data
        data = load_data()
        print(f"✅ Loaded {len(data)} items from dataset")
        
        # Generate vectors
        vectors = prepare_vectors(data)
        print(f"✅ Generated {len(vectors)} embeddings")
        
        # Upload to Pinecone
        upload_to_pinecone(vectors)
        print("✅ Successfully uploaded all embeddings to Pinecone!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 