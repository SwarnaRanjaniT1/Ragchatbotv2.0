import os
import re
import nltk
import PyPDF2
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document processor with embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
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
        
        # Ensure directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            return documents
            
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            
            if filename.endswith(".pdf"):
                text = self._extract_text_from_pdf(file_path)
                doc_type = "pdf"
            elif filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                doc_type = "txt"
            elif filename.endswith(".csv"):
                # Simple CSV handling - more sophisticated parsing would be needed for real use
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                doc_type = "csv"
            elif filename == "README.md":
                continue
            else:
                # Skip unsupported file types
                continue
                
            # Extract year from filename or content (simple approach)
            year_match = re.search(r'20\d{2}', filename)
            year = year_match.group(0) if year_match else "Unknown"
                
            documents.append({
                "content": text,
                "metadata": {
                    "source": filename,
                    "year": year,
                    "type": doc_type
                }
            })
            
        return documents
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks for processing.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(content)
            
            # Create chunk documents with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": i,
                    }
                })
                
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Create embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (chunks with embeddings, embeddings array)
        """
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
            
        return chunks, embeddings
