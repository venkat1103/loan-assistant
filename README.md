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
python fastapi_backend.py
```

4. Start the Streamlit frontend:
```bash
cd llm_model
streamlit run streamlit_frontend.py
```

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
