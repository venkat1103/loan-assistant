from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_5FJVfR_3D7oiX6nAMi9YKCAxJS1mnqmjnsS7fDJQjyqg9B91iWuC19CtLoeMJjK9DGERwB")
index = pc.Index("loan-ai-index")

# Load the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_data(file_path="loan_dataset.jsonl"):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def prepare_vectors(data):
    vectors = []
    for idx, item in enumerate(data):
        # Combine context and question for embedding
        text_to_embed = f"{item['context']} {item['question']}"
        
        # Generate embedding
        encoded_input = tokenizer(text_to_embed, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embedding = embeddings[0].tolist()
        
        vector_id = f"vec_{idx}"
        metadata = {
            "text": text_to_embed,
            "user": item.get("user", ""),
            "question": item.get("question", ""),
            "context": item.get("context", "")
        }
        vectors.append((vector_id, embedding, metadata))
    return vectors

def main():
    try:
        print("🚀 Starting data processing and embedding generation...")
        
        # Load and prepare data
        data = load_data()
        vectors = prepare_vectors(data)
        print(f"✅ Generated embeddings for {len(vectors)} items")
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
            print(f"📦 Uploaded batch {i//batch_size + 1}")
            
        print("🎉 All data processed and uploaded successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 