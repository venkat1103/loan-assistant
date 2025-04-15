from pymongo import MongoClient
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize MongoDB
mongo_client = MongoClient(config['mongodb_uri'])
db = mongo_client['loan_assistant']
questions_collection = db['questions']

def display_stored_data():
    print("\n📊 Data stored in MongoDB:")
    print("=" * 50)
    
    # Get all documents
    documents = questions_collection.find()
    
    # Display each document
    for doc in documents:
        print(f"\nCategory: {doc['category']}")
        print(f"User: {doc['user']}")
        print(f"Question: {doc['question']}")
        print(f"Answer: {doc['answer']}")
        print("Metadata:")
        for key, value in doc['metadata'].items():
            print(f"  - {key}: {value}")
        print("-" * 50)

if __name__ == "__main__":
    display_stored_data() 