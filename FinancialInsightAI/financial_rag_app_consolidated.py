"""
Financial RAG Application - Consolidated Single File

This file contains all components of the financial RAG system:
- Document Processing
- Hybrid Search
- Input/Output Guardrails
- RAG System Core
- Streamlit UI

Author: Replit AI Developer
Date: March 14, 2025
"""

import os
import re
import time
import numpy as np
import streamlit as st
import torch
import nltk
import PyPDF2
from typing import List, Dict, Any, Tuple, Set, Union, Optional
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')

print("NLTK resources setup complete.")

#
# DOCUMENT PROCESSING
#
class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document processor with embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def load_documents(self, data_dir: str = "data") -> List[Dict[str, Any]]:
        """
        Load documents from the data directory.
        
        Args:
            data_dir: Directory containing financial documents
            
        Returns:
            List of document dictionaries with metadata
        """
        documents = []
        
        # Also check for documents in attached_assets directory
        additional_dirs = ["attached_assets"]
        all_dirs = [data_dir] + additional_dirs
        
        for dir_path in all_dirs:
            if not os.path.exists(dir_path):
                continue
                
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Process based on file extension
                if filename.endswith('.pdf'):
                    text = self._extract_text_from_pdf(file_path)
                    if text:
                        # Extract year from filename or content
                        year = self._extract_year(filename, text)
                        documents.append({
                            "content": text,
                            "metadata": {
                                "source": filename,
                                "year": year,
                                "type": "pdf"
                            }
                        })
                elif filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    # Extract year from filename or content
                    year = self._extract_year(filename, text)
                    documents.append({
                        "content": text,
                        "metadata": {
                            "source": filename,
                            "year": year,
                            "type": "text"
                        }
                    })
                    
        print(f"Loaded {len(documents)} documents from {all_dirs}")
        return documents
    
    def _extract_year(self, filename: str, content: str) -> str:
        """
        Extract year from filename or content.
        
        Args:
            filename: Name of the file
            content: Content of the document
            
        Returns:
            Extracted year or "Unknown"
        """
        # First try to find year in filename
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            return year_match.group(0)
        
        # Then try to find year in the first 1000 characters of content
        year_match = re.search(r'(Financial Year|FY|Year)[\s:]*20\d{2}', content[:1000])
        if year_match:
            return re.search(r'20\d{2}', year_match.group(0)).group(0)
            
        # Default to unknown
        return "Unknown"
        
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                return text
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return ""
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks for processing.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries with metadata
        """
        all_chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(content)
            
            # Create chunk objects with metadata
            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": i
                    }
                })
                
        return all_chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Create embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (chunks with embeddings, embeddings array)
        """
        # Extract text content from chunks
        texts = [chunk["content"] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Store embeddings in chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
            
        return chunks, embeddings

#
# HYBRID SEARCH
#

# Helper function for tokenization
def simple_tokenize(text):
    """
    Simple tokenizer that doesn't rely on NLTK's word_tokenize
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    # Remove punctuation and split by whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.split()

class HybridSearch:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, embedding_model: Any):
        """
        Initialize hybrid search with both sparse (BM25) and dense (embedding) retrieval.
        
        Args:
            chunks: List of document chunks
            embeddings: Array of document embeddings
            embedding_model: Model for creating query embeddings
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        
        # Prepare BM25
        # Use simple tokenizer instead of nltk.word_tokenize to avoid issues
        tokenized_chunks = [simple_tokenize(chunk["content"].lower()) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both BM25 and embedding similarity.
        
        Args:
            query: User query
            k: Number of results to return
            alpha: Weight for blending sparse and dense scores (0 = BM25 only, 1 = embeddings only)
            
        Returns:
            List of top k results with scores
        """
        # Get dense scores (embedding similarity)
        query_embedding = self.embedding_model.encode(query)
        dense_scores = self._get_dense_scores(query_embedding)
        
        # Get sparse scores (BM25)
        # Use same simple tokenizer as we used for chunks
        tokenized_query = simple_tokenize(query.lower())
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize scores
        if sparse_scores.max() > 0:
            sparse_scores = sparse_scores / sparse_scores.max()
        if dense_scores.max() > 0:
            dense_scores = dense_scores / dense_scores.max()
        
        # Combine scores
        hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        
        # Get top k results
        top_indices = np.argsort(hybrid_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": float(hybrid_scores[idx]),
                "bm25_score": float(sparse_scores[idx]),
                "embedding_score": float(dense_scores[idx])
            })
            
        return results
    
    def _get_dense_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and all documents.
        
        Args:
            query_embedding: Embedding vector of the query
            
        Returns:
            Array of similarity scores
        """
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        
        return similarities
    
    def rerank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rerank results based on additional criteria (e.g., recency, relevance to specific terms).
        
        Args:
            results: Initial search results
            query: Original query
            
        Returns:
            Reranked results
        """
        # Boost scores for chunks containing specific financial terms
        financial_terms = ["revenue", "profit", "earnings", "growth", "loss", 
                          "assets", "liabilities", "income", "statement", "balance", 
                          "cash flow", "dividend", "equity", "debt", "ratio", 
                          "net income", "operating profit", "EBITDA", "margin"]
        
        # Check if this is a comparison question between years
        query_lower = query.lower()
        is_comparison = any(term in query_lower for term in 
                           ["compare", "comparison", "difference", "change", 
                            "changed", "versus", "vs", "between", "from", "to",
                            "increased", "decreased", "grew", "growth", "reduced"])
                            
        mentioned_years = []
        for year in ["2020", "2021", "2022", "2023", "2024", "2025"]:
            if year in query:
                mentioned_years.append(year)
                
        # Check for financial term matches
        for result in results:
            content = result["content"].lower()
            term_matches = sum(term in content for term in financial_terms)
            
            # Boost based on term matches (small factor)
            boost = 0.05 * term_matches
            
            # Boost recent documents (assuming year is present in metadata)
            year = result["metadata"].get("year", "Unknown")
            
            # Special handling for comparison questions
            if is_comparison and mentioned_years and len(mentioned_years) >= 2:
                # Boost documents that contain the specific years mentioned in query
                if year in mentioned_years:
                    boost += 0.5  # Significant boost for matching years in comparison questions
            elif is_comparison:
                # For generic comparison questions, prioritize content with numeric values
                has_numbers = bool(re.search(r'\d', content))
                if has_numbers:
                    boost += 0.3
                # Prefer chunks that mention multiple years
                years_in_content = sum(y in content for y in ["2020", "2021", "2022", "2023"])
                if years_in_content > 1:
                    boost += 0.4
            else:
                # For non-comparison questions, boost by year if present
                if year != "Unknown":
                    try:
                        # More recent documents get higher boost
                        year_boost = 0.1 * (int(year) - 2018) / 5  # Normalize to recent years
                        boost += max(0, year_boost)  # Ensure non-negative
                    except ValueError:
                        pass
            
            # Apply boost to score
            result["score"] = result["score"] * (1 + boost)
        
        # Re-sort based on adjusted scores
        reranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        return reranked_results

#
# GUARDRAILS
#
class InputGuardrail:
    def __init__(self):
        """
        Initialize input guardrail with common patterns.
        """
        # Financial terms to identify relevant queries
        self.financial_terms = {
            "revenue", "profit", "earnings", "growth", "statement", "income", 
            "financial", "report", "annual", "quarterly", "balance", "sheet", 
            "cash flow", "dividend", "equity", "debt", "ratio", "margin", 
            "expense", "cost", "asset", "liability", "stake", "stock", 
            "shareholder", "investment", "capital", "tax", "fiscal", "budget",
            "ebitda", "eps", "roi", "performance", "sales", "income"
        }
        
        # Patterns for non-financial questions
        self.non_financial_patterns = [
            r"(?:what|where|who|when) is (?!.*?financial)(?!.*?revenue)(?!.*?profit)",
            r"(?:how to|how do|can you|could you) (?!.*?financial)(?!.*?revenue)(?!.*?profit)",
            r"(?:tell me about|explain|describe) (?!.*?financial)(?!.*?revenue)(?!.*?profit)"
        ]
        
        # Explicitly harmful patterns
        self.harmful_patterns = [
            r"how (?:can|to) (?:hack|steal|defraud|manipulate|falsify)",
            r"(?:illegal|unethical) (?:ways|methods|strategies) to",
            r"bypass (?:security|authentication|verification)",
            r"(?:create|generate) (?:fake|false) (?:financial|accounting)"
        ]
        
    def validate_query(self, query: str) -> Tuple[bool, str, float]:
        """
        Validate if a query is appropriate and relevant to financial data.
        
        Args:
            query: User input query
            
        Returns:
            Tuple of (is_valid, reason, confidence)
        """
        query_lower = query.lower()
        
        # Check for harmful patterns
        for pattern in self.harmful_patterns:
            if re.search(pattern, query_lower):
                return (False, "Query appears to request potentially harmful information.", 0.95)
        
        # Check if query is too short
        if len(query_lower.split()) < 3:
            return (False, "Query is too short. Please provide more details.", 0.8)
        
        # Check for financial terms
        financial_term_count = sum(1 for term in self.financial_terms if term in query_lower)
        
        # Check for non-financial patterns
        non_financial_match = any(re.search(pattern, query_lower) for pattern in self.non_financial_patterns)
        
        # Calculate relevance confidence
        if financial_term_count >= 2:
            confidence = min(0.9, 0.5 + 0.1 * financial_term_count)
            return (True, "Query is relevant to financial information.", confidence)
        elif financial_term_count == 1 and not non_financial_match:
            return (True, "Query may be relevant to financial information.", 0.6)
        elif non_financial_match:
            return (False, "Query appears to be unrelated to financial statements.", 0.7)
        else:
            return (False, "Unable to determine if query is related to financial information.", 0.5)

class OutputGuardrail:
    def __init__(self):
        """
        Initialize output guardrail to prevent hallucination or misleading information.
        """
        # Uncertain statement patterns
        self.uncertainty_patterns = [
            r"I (?:think|believe|guess|assume)",
            r"(?:probably|possibly|maybe|perhaps)",
            r"(?:might|may|could) (?:be|have)",
            r"It (?:seems|appears|looks like)"
        ]
        
        # Phrases indicating lack of information
        self.no_info_phrases = [
            "I don't have enough information",
            "I cannot find",
            "There is no information about",
            "The data doesn't include",
            "This information is not available",
            "I don't know"
        ]
    
    def validate_output(self, response: str, context_chunks: List[Dict], query: str) -> Tuple[str, float]:
        """
        Validate and potentially modify the generated response.
        
        Args:
            response: Generated model response
            context_chunks: Retrieved context chunks used for generation
            query: Original user query
            
        Returns:
            Tuple of (validated_response, confidence_score)
        """
        query_lower = query.lower()
        
        # Check if response contains uncertainty patterns
        uncertainty_found = any(re.search(pattern, response.lower()) for pattern in self.uncertainty_patterns)
        
        # Check if the response states lack of information
        no_info_found = any(phrase.lower() in response.lower() for phrase in self.no_info_phrases)
        
        # Check if query is asking for a comparison between years
        is_comparison = any(term in query_lower for term in 
                          ["compare", "comparison", "difference", "change", 
                           "changed", "versus", "vs", "between", "from", "to",
                           "increased", "decreased", "grew", "growth", "reduced"])
        
        # For comparison questions, verify that the response contains numbers
        numeric_pattern = r'\d+([,.]\d+)?'
        contains_numbers = bool(re.search(numeric_pattern, response))
        
        # Identify mentioned years in query
        mentioned_years = [year for year in ["2020", "2021", "2022", "2023"] if year in query]
        
        # Calculate base confidence score based on retrieval scores
        if context_chunks:
            base_confidence = sum(chunk.get('score', 0) for chunk in context_chunks) / len(context_chunks)
        else:
            base_confidence = 0.1
            
        # Adjust confidence based on various signals
        confidence = base_confidence
        
        if uncertainty_found:
            confidence *= 0.7
            
        if no_info_found:
            confidence *= 0.5
            
        # Additional validation for comparison questions
        if is_comparison:
            # Verify the answer contains numbers for comparison questions
            if not contains_numbers:
                confidence *= 0.4
                disclaimer = ("\n\nNote: This answer may not provide specific numeric comparisons "
                             "as requested. Please check the provided context for details.")
                response += disclaimer
            
            # For year comparisons, verify mentioned years appear in response
            if mentioned_years:
                years_in_response = sum(year in response for year in mentioned_years)
                # Reduce confidence if years are missing
                if years_in_response < len(mentioned_years):
                    confidence *= 0.6
                    
                    # Enhance response with specific year information if missing
                    if confidence < 0.5:
                        # Check if context contains missing years 
                        context_text = " ".join([chunk.get("content", "") for chunk in context_chunks])
                        missing_years_info = []
                        for year in mentioned_years:
                            if year not in response and year in context_text:
                                # Add missing info to disclaimer
                                missing_years_info.append(f"Year {year} is mentioned in the context but not in the response.")
                                
                        if missing_years_info:
                            response += "\n\nAdditional information: " + " ".join(missing_years_info)
            
        # Add disclaimer for low confidence responses
        if confidence < 0.5 and not "Note:" in response:
            disclaimer = ("\n\nNote: This answer has low confidence based on the available "
                         "financial data. Please verify with additional sources.")
            response += disclaimer
            
        # Cap confidence
        confidence = min(0.95, confidence)
        
        return response, confidence

#
# RAG SYSTEM
#
class FinancialRAG:
    def __init__(self):
        """
        Initialize the Financial RAG system with all components.
        """
        # Initialize the document processor
        self.doc_processor = DocumentProcessor()
        
        # Initialize the guardrails
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        
        # Load and process documents
        self.chunks = []
        self.embeddings = None
        self.hybrid_search = None
        self.initialize_document_store()
        
        # Load language model for generation
        self.initialize_language_model()
        
    def initialize_document_store(self):
        """
        Load, process, and index all financial documents.
        """
        # Load documents from the data directory
        documents = self.doc_processor.load_documents()
        
        if not documents:
            print("Warning: No documents found in the data directory.")
            return
            
        # Process and chunk documents
        self.chunks = self.doc_processor.chunk_documents(documents)
        
        # Create embeddings
        self.chunks, self.embeddings = self.doc_processor.create_embeddings(self.chunks)
        
        # Initialize hybrid search
        self.hybrid_search = HybridSearch(
            self.chunks, 
            self.embeddings, 
            self.doc_processor.embedding_model
        )
        
        print(f"Initialized document store with {len(self.chunks)} chunks from {len(documents)} documents.")
        
    def initialize_language_model(self):
        """
        Initialize the small language model for text generation.
        """
        # Using FLAN-T5-small for generation (a relatively small model)
        model_name = "google/flan-t5-small"
        
        # Set device to CPU explicitly to avoid CUDA issues
        device = "cpu"
        print(f"Device set to use {device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Configure generation pipeline with device set to CPU
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            device=device
        )
        
        print(f"Initialized language model: {model_name}")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Process the user question and generate an answer.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer, confidence, and supporting context
        """
        # Apply input guardrail
        is_valid, reason, validation_confidence = self.input_guardrail.validate_query(question)
        
        if not is_valid:
            return {
                "answer": f"I cannot answer this question because: {reason}",
                "confidence": validation_confidence,
                "context": [],
                "is_valid": False
            }
        
        # If no documents are loaded
        if not self.hybrid_search:
            return {
                "answer": "No financial documents have been loaded. Please add financial statements to the data directory.",
                "confidence": 0.0,
                "context": [],
                "is_valid": True
            }
        
        # Retrieve relevant chunks using hybrid search
        # Increase k to get more potential matches
        retrieved_chunks = self.hybrid_search.search(question, k=8, alpha=0.7)
        
        # Rerank results
        reranked_chunks = self.hybrid_search.rerank(retrieved_chunks, question)
        
        # Filter to top 5 most relevant chunks
        reranked_chunks = reranked_chunks[:5]
        
        # Format context for the language model
        context_text = "\n\n".join([f"Document {i+1}: {chunk['content']}" 
                                  for i, chunk in enumerate(reranked_chunks)])
        
        # Prepare prompt for the language model
        prompt = f"""
Answer the following question about financial information based ONLY on the provided context.
If the context doesn't contain the information to answer the question, say "I don't have enough information to answer this question based on the provided financial data."
Be precise and specific in your answer, focusing only on financial metrics, numbers, and facts directly relevant to the question.
For questions about changes or comparisons between years, provide specific numbers from the financial statements.

Context:
{context_text}

Question: {question}

Answer:
"""
        
        # Generate answer
        generated_output = self.generator(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']
        
        # Apply output guardrail
        validated_answer, confidence = self.output_guardrail.validate_output(
            generated_output, 
            reranked_chunks, 
            question
        )
        
        # Prepare context information for display
        context_info = []
        for chunk in reranked_chunks:
            context_info.append({
                "content": chunk["content"],
                "source": chunk["metadata"].get("source", "Unknown"),
                "year": chunk["metadata"].get("year", "Unknown"),
                "score": chunk["score"]
            })
        
        return {
            "answer": validated_answer,
            "confidence": confidence,
            "context": context_info,
            "is_valid": True
        }

#
# STREAMLIT UI
#
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
            
            # Create data directory if it doesn't exist
            if not os.path.exists("data"):
                os.makedirs("data")
                
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
        - How did net income change between 2022 and 2023?
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

# Run the app
if __name__ == "__main__":
    main()