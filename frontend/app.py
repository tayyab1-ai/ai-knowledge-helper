import streamlit as st
import requests
import json

# --- Page Config ---
st.set_page_config(
    page_title="RAG AI Knowledge Helper",
    page_icon="🤖",
    layout="centered"
)

# --- Custom CSS for Chatbot UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #e1f5fe;
    }
    .stChatMessage.assistant {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .source-tag {
        font-size: 0.8rem;
        color: #666;
        background-color: #eee;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000/query")
    st.divider()
    st.markdown("""
    ### About
    This is a **Cluster-Aware RAG** system designed to help with Pakistani banking queries.
    
    **Features:**
    - FastAPI Backend
    - Streamlit Frontend
    - FAISS Vector Search
    - Cluster-based Retrieval
    - Llama 3.1 8B (Groq)
    """)
    
    if st.button("Check API Health"):
        try:
            health_url = api_url.replace("/query", "/health")
            response = requests.get(health_url)
            if response.status_code == 200:
                st.success("API is Online!")
                st.json(response.json())
            else:
                st.error(f"API returned status: {response.status_code}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")

# --- Main UI ---
st.title("🤖 AI Knowledge Helper")
st.caption("Ask me anything about Pakistani banking services, accounts, or regulations.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.markdown("**Sources:** " + " ".join([f'<span class="source-tag">{s}</span>' for s in message["sources"]]), unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call FastAPI backend
            response = requests.post(api_url, json={"question": prompt})
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                
                # Update UI
                message_placeholder.markdown(answer)
                if sources:
                    st.markdown("**Sources:** " + " ".join([f'<span class="source-tag">{s}</span>' for s in sources]), unsafe_allow_html=True)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            else:
                error_msg = f"Error: API returned status {response.status_code}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_msg = f"Error: Could not connect to backend. {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("Built with ❤️ for Pakistani Banking Customers")
