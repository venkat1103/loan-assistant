# Loan Assistant

An AI-powered loan assistant that helps users get information about their loans, EMIs, and related queries using natural language processing.

## Features

- Natural language query processing for loan-related questions
- Integration with MongoDB for data storage
- Vector similarity search using Pinecone
- Real-time EMI calculations and loan status updates
- User-specific loan information tracking

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with:
```
MONGODB_URI=your_mongodb_uri
PINECONE_API_KEY=your_pinecone_api_key
```

3. Start the FastAPI backend:
```bash
cd llm_model
python -m uvicorn fastapi_backend:app --host 0.0.0.0 --port 8000
```

4. Start the Streamlit frontend:
```bash
cd llm_model
streamlit run streamlit_frontend.py
```

## Deployment

### MongoDB Atlas Setup
1. Create a MongoDB Atlas account and cluster
2. Set up a database user with read/write permissions
3. Configure Network Access to allow connections from your deployment
4. Get your connection string in the format: `mongodb+srv://username:password@cluster.mongodb.net/loan_assistant`

### Render Deployment
1. Create a new Web Service in Render
2. Connect to your GitHub repository
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `cd llm_model && uvicorn fastapi_backend:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `MONGODB_URI`: Your MongoDB Atlas connection string
   - `PINECONE_API_KEY`: Your Pinecone API key

### Streamlit Deployment
1. For Streamlit Cloud deployment, use the simplified requirements:
   - Create a new app in Streamlit Cloud
   - Point to your GitHub repo and the main.py file
   - Set the `api_url` secret to your Render backend URL

2. For local Streamlit with cloud backend:
   - Create a `.streamlit/secrets.toml` file based on the example
   - Set `api_url` to your Render backend URL
   - Run streamlit locally: `cd llm_model && streamlit run streamlit_frontend.py`

## Usage

1. Select a user from the dropdown menu
2. Type your loan-related question in the text input
3. Click "Ask" to get relevant answers

## Example Questions

- What is my loan interest rate?
- When is my next EMI due?
- How much of my loan is pending?
- How much interest have I paid so far?
- What are my prepayment benefits? # loan-assistant
