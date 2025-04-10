"""
Streamlit app for the Document Assistant.
"""
import os
import time
import tempfile
import streamlit as st
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingsManager
from src.llm_manager import LLMManager
from src.qa_engine import QAEngine

# App title and description
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Assistant")
st.markdown("""
This application analyzes contracts and provides a Q&A interface to interact with the document content.
Upload a PDF contract, and then ask questions about it.
""")

# Initialization function
@st.cache_resource
def initialize_qa_engine(retrieval_method="vector", model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Initialize the QA engine with caching to avoid reloading the model."""
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Initialize managers
    embeddings_manager = EmbeddingsManager()
    
    # Show more detailed loading message
    with st.spinner(f"Loading LLM model ({model_name})... This might take a few minutes."):
        llm_manager = LLMManager(model_name=model_name)
    
    # Create QA engine
    qa_engine = QAEngine(
        embeddings_manager=embeddings_manager,
        llm_manager=llm_manager,
        retrieval_method=retrieval_method
    )
    
    return qa_engine

# Initialize session state for storing document info
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False

if 'document_name' not in st.session_state:
    st.session_state.document_name = ""

if 'document_summary' not in st.session_state:
    st.session_state.document_summary = ""

if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Sidebar for configuration and document upload
with st.sidebar:
    st.header("Configuration")
    
    # Model selection (simplified for now)
    llm_model = st.selectbox(
        "LLM Model",
        ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TheBloke/Llama-2-7B-Chat-GGUF"]
    )
    
    # Architecture ablation 1: Chunking strategy
    chunking_strategy = st.radio(
        "Chunking Strategy",
        ["fixed", "semantic"],
        help="Fixed: Split by token count. Semantic: Split by logical sections."
    )
    
    # Architecture ablation 2: Retrieval method
    retrieval_method = st.radio(
        "Retrieval Method",
        ["vector", "keyword"],
        help="Vector: Semantic similarity search. Keyword: Key term matching."
    )
    
    st.divider()
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a contract (PDF)", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        # Process button
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Initialize QA engine
                qa_engine = initialize_qa_engine(
                    retrieval_method=retrieval_method,
                    model_name=llm_model
                )
                
                # Load the document
                qa_engine.load_document(
                    pdf_path,
                    chunk_strategy=chunking_strategy,
                    save_dir="data/processed"
                )
                
                # Generate summary
                with st.spinner("Generating document summary..."):
                    summary = qa_engine.generate_document_summary(max_words=100)
                    st.session_state.document_summary = summary
                
                # Update session state
                st.session_state.document_loaded = True
                st.session_state.document_name = uploaded_file.name
                st.session_state.qa_engine = qa_engine
                st.session_state.qa_history = []
                
                # Cleanup temp file
                os.unlink(pdf_path)
                
                st.success(f"Document processed successfully: {uploaded_file.name}")
                st.balloons()

# Main content area
if st.session_state.document_loaded:
    # Document info
    st.header(f"Document: {st.session_state.document_name}")
    
    # Display document summary
    with st.expander("Document Summary", expanded=True):
        st.markdown(st.session_state.document_summary)
    
    # Q&A interface
    st.header("Ask Questions")
    
    # Question input
    question = st.text_input("Enter your question about the document:")
    
    if st.button("Ask"):
        if question:
            with st.spinner("Generating answer..."):
                # Get answer from QA engine
                qa_engine = st.session_state.qa_engine
                result = qa_engine.answer_question(question)
                
                # Add to history
                st.session_state.qa_history.append(result)
                
                # Force refresh
                st.experimental_rerun()
    
    # Display Q&A history
    if st.session_state.qa_history:
        st.header("Q&A History")
        
        for i, item in enumerate(st.session_state.qa_history):
            with st.container():
                st.subheader(f"Q: {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                
                with st.expander("Show relevant context"):
                    st.markdown(item['context'])
                
                if i < len(st.session_state.qa_history) - 1:
                    st.divider()
else:
    # Instructions for when no document is loaded
    st.info("Please upload a contract document (PDF) and click 'Process Document' to begin.")
    
    # Example questions
    st.header("Example Questions You Can Ask")
    example_questions = [
        "What are the payment terms in this contract?",
        "When does this contract expire?",
        "What are the termination conditions?",
        "Who are the parties involved in this agreement?",
        "What are the confidentiality obligations?"
    ]
    
    for question in example_questions:
        st.markdown(f"- {question}")

# Footer
st.divider()
st.markdown("*welcome to the document assistance system.*")