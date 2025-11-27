import streamlit as st
import os
import pickle
import time
import faiss
from sentence_transformers import SentenceTransformer

from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import CTransformers
import tempfile
import requests

# Page configuration
st.set_page_config(
    page_title="E-commerce Customer Support Chatbot",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    .bot-message {
        background-color: #e8f4f8;
        border-left-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_resource
def download_hugging_face_embeddings():
    """Download and cache Hugging Face embeddings model"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

@st.cache_resource
def load_sentence_transformer():
    """Load and cache SentenceTransformer model"""
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

def load_data_from_url(url):
    """Load and process data from URL"""
    try:
        with st.spinner("Loading data from URL..."):
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()
            
            # Split the documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=20
            )
            docs = text_splitter.split_documents(data)
            
            return docs
    except Exception as e:
        st.error(f"Error loading data from URL: {str(e)}")
        return None

def create_vectorstore(docs):
    """Create FAISS vectorstore from documents"""
    try:
        with st.spinner("Creating vector embeddings..."):
            embeddings = download_hugging_face_embeddings()
            vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
            return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def initialize_llm():
    """Initialize the language model (using a lightweight alternative)"""
    try:
        # For demonstration purposes, we'll use HuggingFace transformers
        # You can replace this with your preferred LLM
        from transformers import pipeline
        
        # Use a lightweight model for demonstration
        qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_length=200,
            temperature=0.7
        )
        return qa_pipeline
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def setup_qa_chain(vectorstore):
    """Set up the QA chain with retriever"""
    try:
        if vectorstore is None:
            return None
        
        # Create a simple retriever-based system
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"Error setting up QA chain: {str(e)}")
        return None

def get_answer(query, retriever, qa_pipeline):
    """Get answer using retriever and QA pipeline"""
    try:
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "I couldn't find relevant information to answer your question."
        
        # Combine retrieved documents
        context = "\n".join([doc.page_content for doc in docs[:3]])
        
        # Create prompt for the model
        prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate answer using the pipeline
        response = qa_pipeline(prompt)
        
        if isinstance(response, list) and len(response) > 0:
            answer = response[0].get('generated_text', 'No answer generated.')
        else:
            answer = "I couldn't generate an answer."
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Main application
def main():
    st.markdown("<h1 class='main-header'>🛒 E-commerce Customer Support Chatbot</h1>", unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # URL input
        default_url = "https://www.mcentre.lk/store/categories/special-offers?price-max=66096&price-min=39900&type%5B0%5D=students-offer&type%5B1%5D=teachers-offers"
        url = st.text_input("Enter URL to load data from:", value=default_url)
        
        if st.button("Load Data"):
            if url:
                docs = load_data_from_url(url)
                if docs:
                    st.session_state.vectorstore = create_vectorstore(docs)
                    if st.session_state.vectorstore:
                        st.session_state.qa_chain = setup_qa_chain(st.session_state.vectorstore)
                        st.session_state.data_loaded = True
                        st.success("Data loaded successfully!")
                    else:
                        st.error("Failed to create vectorstore")
                else:
                    st.error("Failed to load data from URL")
            else:
                st.error("Please enter a valid URL")
        
        # Data status
        if st.session_state.data_loaded:
            st.success("✅ Data loaded and ready")
        else:
            st.warning("⚠️ No data loaded")
        
        # Instructions
        st.markdown("""
        ### Instructions:
        1. Load data from a URL using the form above
        2. Ask questions about the loaded content
        3. The chatbot will provide answers based on the data
        
        ### Sample Questions:
        - What products are available?
        - What are the prices?
        - What special offers are there?
        """)
    
    # Main chat interface
    st.header("Chat with Customer Support")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Bot:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about our products..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.container():
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {prompt}
            </div>
            """, unsafe_allow_html=True)
        
        # Generate response
        if st.session_state.data_loaded and st.session_state.qa_chain:
            with st.spinner("Thinking..."):
                # Initialize QA pipeline if not exists
                if 'qa_pipeline' not in st.session_state:
                    st.session_state.qa_pipeline = initialize_llm()
                
                if st.session_state.qa_pipeline:
                    response = get_answer(prompt, st.session_state.qa_chain, st.session_state.qa_pipeline)
                else:
                    response = "I'm sorry, the language model is not available. Please try again later."
        else:
            response = "Please load data first using the sidebar to get started!"
        
        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display bot response
        with st.container():
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Bot:</strong> {response}
            </div>
            """, unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
