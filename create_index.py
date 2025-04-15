from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone with your API key
pc = Pinecone(api_key="pcsk_5FJVfR_3D7oiX6nAMi9YKCAxJS1mnqmjnsS7fDJQjyqg9B91iWuC19CtLoeMJjK9DGERwB")

# Define index name
index_name = "loan-ai-index"

# Check if index exists
if index_name not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=768,  # This should match the dimension of your embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index to verify it's working
try:
    index = pc.Index(index_name)
    print("✅ Successfully connected to the index!")
except Exception as e:
    print(f"❌ Error connecting to index: {str(e)}") 