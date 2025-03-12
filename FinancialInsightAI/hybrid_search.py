from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from sklearn.preprocessing import normalize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

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
        tokenized_chunks = [nltk.word_tokenize(chunk["content"].lower()) for chunk in chunks]
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
        tokenized_query = nltk.word_tokenize(query.lower())
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
                          "cash flow", "dividend", "equity", "debt", "ratio"]
        
        # Check for financial term matches
        for result in results:
            content = result["content"].lower()
            term_matches = sum(term in content for term in financial_terms)
            
            # Boost based on term matches (small factor)
            boost = 0.05 * term_matches
            
            # Boost recent documents (assuming year is present in metadata)
            year = result["metadata"].get("year", "Unknown")
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
