#!/usr/bin/env python3
import pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import sys

def show_status(message, status):
    """Helper function to display colored status messages"""
    status_colors = {
        "success": "\033[92m",  # Green
        "error": "\033[91m",    # Red
        "warning": "\033[93m",  # Yellow
        "info": "\033[94m",     # Blue
        "end": "\033[0m"        # Reset
    }
    print(f"{status_colors.get(status, '')}▶ {message}{status_colors['end']}")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    # Startup banner
    print("\n" + "="*50)
    show_status("🏦 Loan Question Answering System - STARTING", "info")
    print("="*50)
    
    # 1. Initialize Pinecone
    show_status("Initializing Pinecone connection...", "info")
    try:
        pc = pinecone.Pinecone(api_key="pcsk_5FJVfR_3D7oiX6nAMi9YKCAxJS1mnqmjnsS7fDJQjyqg9B91iWuC19CtLoeMJjK9DGERwB")
        index = pc.Index("loan-ai-index")
        show_status("✓ Pinecone connection established", "success")
    except Exception as e:
        show_status(f"✗ Pinecone initialization failed: {str(e)}", "error")
        sys.exit(1)

    # 2. Load embedding model
    show_status("Loading embedding model...", "info")
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        show_status("✓ Embedding model loaded successfully", "success")
    except Exception as e:
        show_status(f"✗ Model loading failed: {str(e)}", "error")
        sys.exit(1)

    # 3. System ready message
    show_status("SYSTEM READY - You can now ask questions", "success")
    print("- Type 'quit' to exit")
    print("="*50)

    # 4. Query interface
    while True:
        try:
            query = input("\n❓ Question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                show_status("Shutting down system...", "info")
                break
                
            # Process query
            show_status("Processing question...", "info")
            encoded_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Perform pooling
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embedding = embeddings[0].tolist()
            
            show_status("Searching knowledge base...", "info")
            results = index.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )
            
            if results.matches:
                show_status(f"Found {len(results.matches)} relevant answers:", "success")
                for i, match in enumerate(results.matches, 1):
                    print(f"\n⭐ Answer {i} (Confidence: {match.score:.2%})")
                    print(f"Q: {match.metadata['question']}")
                    print(f"A: {match.metadata['answer']}")
            else:
                show_status("No matching answers found", "warning")
                
        except KeyboardInterrupt:
            show_status("\nSystem shutdown requested", "warning")
            break
        except Exception as e:
            show_status(f"Error: {str(e)}", "error")

    # Shutdown message
    show_status("System shutdown complete", "info")
    print("="*50)

if __name__ == "__main__":
    main()