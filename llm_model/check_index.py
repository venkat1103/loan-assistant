import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone client
pc = Pinecone(
    api_key="pcsk_5FJVfR_3D7oiX6nAMi9YKCAxJS1mnqmjnsS7fDJQjyqg9B91iWuC19CtLoeMJjK9DGERwB"  # Replace with your actual Pinecone API key
)

# Create the serverless index if it doesn't exist
if 'loan-ai-index' not in pc.list_indexes().names():
    pc.create_index(
        name='loan-ai-index',
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
    print("Index created successfully!")
else:
    print("Index already exists!")