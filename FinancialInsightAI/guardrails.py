import re
from typing import Dict, Union, Tuple, List, Set

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
        # Check if response contains uncertainty patterns
        uncertainty_found = any(re.search(pattern, response.lower()) for pattern in self.uncertainty_patterns)
        
        # Check if the response states lack of information
        no_info_found = any(phrase.lower() in response.lower() for phrase in self.no_info_phrases)
        
        # Calculate base confidence score based on retrieval scores
        if context_chunks:
            base_confidence = sum(chunk.get('score', 0) for chunk in context_chunks) / len(context_chunks)
        else:
            base_confidence = 0.1
            
        # Adjust confidence based on uncertainty and no_info signals
        confidence = base_confidence
        if uncertainty_found:
            confidence *= 0.7
        if no_info_found:
            confidence *= 0.5
            
        # Add disclaimer for low confidence responses
        if confidence < 0.5:
            disclaimer = ("\n\nNote: This answer has low confidence based on the available "
                         "financial data. Please verify with additional sources.")
            response += disclaimer
            
        # Cap confidence
        confidence = min(0.95, confidence)
        
        return response, confidence
