import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    print("WARNING: PINECONE_API_KEY environment variable not set")
    pinecone_api_key = "pcsk_5FJVfR_3D7oiX6nAMi9YKCAxJS1mnqmjnsS7fDJQjyqg9B91iWuC19CtLoeMJjK9DGERwB"

pc = Pinecone(api_key=pinecone_api_key)

# Create the serverless index if it doesn't exist
if 'loan-ai-index' not in pc.list_indexes().names():
    print("Creating new Pinecone index 'loan-ai-index'...")
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
    print("Index 'loan-ai-index' already exists!")