import streamlit as st
import requests
import json
import os

# Configure page
st.set_page_config(
    page_title="Loan Assistant",
    page_icon="💰",
    layout="wide"
)

# API endpoint - use environment variable or default to localhost
API_URL = st.secrets.get("api_url", "http://localhost:8000")

# Display backend connection status
st.sidebar.markdown("### Backend Status")
try:
    response = requests.get(f"{API_URL}/users")
    if response.status_code == 200:
        st.sidebar.success("✅ Connected to backend")
    else:
        st.sidebar.error(f"❌ Backend error: {response.status_code}")
except:
    st.sidebar.error(f"❌ Cannot connect to backend at {API_URL}")
    st.info("⚠️ The backend service is not available. Please make sure the backend URL is correctly configured in Streamlit's secrets.")

def get_users():
    try:
        response = requests.get(f"{API_URL}/users")
        if response.status_code == 200:
            return response.json()["users"]
    except:
        st.error("Unable to connect to backend service")
    return []

def get_categories():
    try:
        response = requests.get(f"{API_URL}/categories")
        if response.status_code == 200:
            return response.json()["categories"]
    except:
        st.error("Unable to connect to backend service")
    return []

def query_assistant(question, selected_user=None):
    try:
        payload = {
            "question": question,
            "user": selected_user
        }
        response = requests.post(f"{API_URL}/query", json=payload)
        if response.status_code == 200:
            return response.json()
    except:
        st.error("Unable to connect to backend service")
    return []

# Sidebar
st.sidebar.title("Loan Assistant")
st.sidebar.markdown("---")

# User selection
users = get_users()
selected_user = st.sidebar.selectbox("Select User", ["All Users"] + users)
if selected_user == "All Users":
    selected_user = None

# Categories display
st.sidebar.markdown("---")
st.sidebar.subheader("Available Categories")
categories = get_categories()
for category in categories:
    st.sidebar.markdown(f"- {category.replace('_', ' ').title()}")

# Main content
st.title("💰 Loan Assistant")
st.markdown("Ask questions about loans, EMIs, and more!")

# Example questions based on categories
st.markdown("### Example Questions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Interest & EMI**")
    st.markdown("- What is Venkat's loan interest rate?")
    st.markdown("- When is Shay's next EMI due?")
    st.markdown("- How much EMI does Karthik need to pay?")

with col2:
    st.markdown("**Loan Status & Prepayment**")
    st.markdown("- How much of Venkat's loan is pending?")
    st.markdown("- How much interest has Shay paid so far?")
    st.markdown("- What's the prepayment benefit for Karthik?")

# Question input
question = st.text_input("Your Question:", placeholder="Type your question here...")

if st.button("Ask"):
    if question:
        with st.spinner("Searching for answers..."):
            answers = query_assistant(question, selected_user)
            
            if answers:
                for i, answer in enumerate(answers, 1):
                    with st.expander(f"Answer {i} (Score: {answer['similarity_score']:.2f} via {answer['source']})"):
                        for match in answer['matches']:
                            st.markdown(f"**Question:** {match['question']}")
                            st.markdown(f"**Answer:** {match['answer']}")
                            st.markdown(f"**User:** {match['user']}")
                            
                            # Display metadata
                            st.markdown("**Additional Information:**")
                            for key, value in match['metadata'].items():
                                if key not in ['question', 'answer', 'user']:
                                    st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
            else:
                st.warning("No relevant answers found. Please try rephrasing your question.")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Team") 