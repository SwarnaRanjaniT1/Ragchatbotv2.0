from typing import Dict, List, Any, Tuple, Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from document_processor import DocumentProcessor
from hybrid_search import HybridSearch
from guardrails import InputGuardrail, OutputGuardrail

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
