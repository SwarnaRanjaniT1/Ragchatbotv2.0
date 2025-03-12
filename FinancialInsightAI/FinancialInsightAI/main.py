import streamlit as st
import os
import time
from rag_system import FinancialRAG

# Set page configuration
st.set_page_config(
    page_title="Financial Q&A",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
    .context-box {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metadata {
        font-size: 0.8rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """
    Load and cache the RAG system.
    """
    return FinancialRAG()

def format_confidence(confidence):
    """
    Format confidence score with appropriate styling.
    """
    if confidence >= 0.7:
        return f'<span class="confidence-high">{confidence:.2f}</span>'
    elif confidence >= 0.4:
        return f'<span class="confidence-medium">{confidence:.2f}</span>'
    else:
        return f'<span class="confidence-low">{confidence:.2f}</span>'

def main():
    # Display header
    st.markdown('<h1 class="main-header">Financial Statement Q&A</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about company financial data</p>', unsafe_allow_html=True)
    
    # Initialize or load the RAG system
    with st.spinner("Loading RAG system..."):
        rag_system = load_rag_system()
    
    # Sidebar for file upload and information
    with st.sidebar:
        st.markdown("### Upload Financial Statements")
        uploaded_files = st.file_uploader(
            "Upload financial statements (PDF, TXT, CSV)", 
            accept_multiple_files=True,
            type=["pdf", "txt", "csv"]
        )
        
        if uploaded_files:
            st.info("Processing uploaded files...")
            
            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
            # Reinitialize document store
            rag_system.initialize_document_store()
            st.success(f"Processed {len(uploaded_files)} files successfully!")
        
        st.markdown("### Sample Questions")
        st.markdown("""
        - What was the revenue in the most recent year?
        - How did net income change between 2021 and 2022?
        - What are the main sources of revenue?
        - What's the company's debt to equity ratio?
        - What was the year-over-year growth in operating income?
        """)
        
        st.markdown("### About")
        st.markdown("""
        This system uses Retrieval-Augmented Generation (RAG) with:
        
        - Hybrid search (BM25 + embeddings)
        - Input validation guardrails
        - Small open-source language model
        - Source attribution
        
        The system only answers based on uploaded financial documents.
        """)
    
    # Main content area
    query = st.text_input("Ask a question about the financial statements:", key="question_input")
    
    if st.button("Submit Question") or query:
        if not query:
            st.warning("Please enter a question.")
        else:
            # Show spinner during processing
            with st.spinner("Analyzing financial data..."):
                # Add a small delay to show the spinner
                time.sleep(0.5)
                # Process the question
                result = rag_system.answer_question(query)
            
            # Display confidence and answer
            conf_display = format_confidence(result["confidence"])
            st.markdown(f"**Confidence Score:** {conf_display}", unsafe_allow_html=True)
            
            # Display answer
            st.markdown("### Answer")
            if result["is_valid"]:
                st.markdown(result["answer"])
            else:
                st.error(result["answer"])
            
            # Display context if available
            if result["context"]:
                st.markdown("### Supporting Context")
                
                for i, ctx in enumerate(result["context"]):
                    with st.expander(f"Source {i+1}: {ctx['source']} ({ctx['year']}) - Relevance: {ctx['score']:.2f}"):
                        st.markdown(ctx["content"])
            elif result["is_valid"]:
                st.info("No context was used to generate this answer.")

if __name__ == "__main__":
    main()
