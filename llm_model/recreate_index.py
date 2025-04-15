from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_5FJVfR_3D7oiX6nAMi9YKCAxJS1mnqmjnsS7fDJQjyqg9B91iWuC19CtLoeMJjK9DGERwB")

# Define index name
index_name = "loan-ai-index"

# Delete the existing index if it exists
if index_name in pc.list_indexes().names():
    print(f"Deleting existing index '{index_name}'...")
    pc.delete_index(index_name)
    print("✅ Index deleted successfully!")

# Create new index with correct dimensions
print(f"Creating new index '{index_name}'...")
pc.create_index(
    name=index_name,
    dimension=384,  # Changed to match all-MiniLM-L6-v2 model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
print("✅ Index created successfully!")

# Verify the index exists
if index_name in pc.list_indexes().names():
    print("✅ Index verification successful!")
else:
    print("❌ Index verification failed!") 