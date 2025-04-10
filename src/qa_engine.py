"""
Q&A Engine module for answering questions about documents.
"""
import os
import re
from typing import List, Dict, Any, Optional
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingsManager
from src.llm_manager import LLMManager

class QAEngine:
    """Handles document Q&A using embeddings and a local LLM."""
    
    def __init__(self, 
                 embeddings_manager: EmbeddingsManager,
                 llm_manager: LLMManager,
                 retrieval_method: str = "vector",
                 top_k: int = 3):
        """
        Initialize the Q&A engine.
        
        Args:
            embeddings_manager: Manager for document embeddings
            llm_manager: Manager for LLM interactions
            retrieval_method: Method for retrieving relevant chunks ('vector' or 'keyword')
            top_k: Number of chunks to retrieve for each question
        """
        self.embeddings_manager = embeddings_manager
        self.llm_manager = llm_manager
        self.retrieval_method = retrieval_method
        self.top_k = top_k
        self.document_text = ""
    
    def load_document(self, document_path: str, 
                     chunk_strategy: str = "fixed",
                     save_dir: str = "data/processed") -> None:
        """
        Load and process a document for Q&A.
        
        Args:
            document_path: Path to the document file
            chunk_strategy: Strategy for chunking ('fixed' or 'semantic')
            save_dir: Directory to save processed data
        """
        # Process the document
        processor = DocumentProcessor(chunk_strategy=chunk_strategy)
        result = processor.process_document(document_path)
        
        # Store the full text
        self.document_text = result["full_text"]
        
        # Process the chunks with the embeddings manager
        self.embeddings_manager.process_chunks(result["chunks"])
        
        # Save the processed data
        os.makedirs(save_dir, exist_ok=True)
        self.embeddings_manager.save(save_dir)
    
    def _retrieve_chunks_vector(self, question: str) -> List[str]:
        """
        Retrieve relevant document chunks using vector similarity.
        
        Args:
            question: The question to find relevant chunks for
            
        Returns:
            List of relevant text chunks
        """
        search_results = self.embeddings_manager.search(question, top_k=self.top_k)
        return [result["chunk"] for result in search_results]
    
    def _retrieve_chunks_keyword(self, question: str) -> List[str]:
        """
        Retrieve relevant document chunks using keyword matching.
        
        Args:
            question: The question to find relevant chunks for
            
        Returns:
            List of relevant text chunks
        """
        # Extract keywords from the question (simple approach)
        # Remove stop words and punctuation
        keywords = re.sub(r'[^\w\s]', '', question.lower())
        keywords = re.sub(r'\b(the|a|an|in|of|on|for|to|with|by|at|from)\b', '', keywords)
        keywords = [k.strip() for k in keywords.split() if len(k.strip()) > 3]
        
        # Score each chunk based on keyword occurrences
        chunks = self.embeddings_manager.chunks
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for k in keywords if k in chunk_lower)
            chunk_scores.append((i, score))
        
        # Sort by score and get top_k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in chunk_scores[:self.top_k] if score > 0]
        
        # Return the top chunks
        return [chunks[idx] for idx in top_indices]
    
    def retrieve_relevant_context(self, question: str) -> str:
        """
        Retrieve relevant context for a question using the selected method.
        
        Args:
            question: The question to find context for
            
        Returns:
            Combined relevant context
        """
        if self.retrieval_method == "vector":
            chunks = self._retrieve_chunks_vector(question)
        else:  # keyword method
            chunks = self._retrieve_chunks_keyword(question)
        
        # Combine chunks into a single context
        context = "\n\n".join(chunks)
        return context
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about the document.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with the answer and relevant context
        """
        # Retrieve relevant context
        context = self.retrieve_relevant_context(question)
        
        # If no context was found, use a portion of the document
        if not context.strip():
            context = self.document_text[:2000] + "..."
        
        # Generate an answer using the LLM
        answer = self.llm_manager.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context": context
        }
    
    def generate_document_summary(self, max_words: int = 100) -> str:
        """
        Generate a summary of the entire document.
        
        Args:
            max_words: Maximum number of words for the summary
            
        Returns:
            Document summary
        """
        # Use the first portion of the document if it's very large
        text_to_summarize = self.document_text
        if len(text_to_summarize) > 10000:
            text_to_summarize = text_to_summarize[:10000] + "..."
        
        # Generate the summary
        summary = self.llm_manager.generate_summary(text_to_summarize, max_words=max_words)
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize managers
    embeddings_manager = EmbeddingsManager()
    llm_manager = LLMManager()
    
    # Create the QA engine
    qa_engine = QAEngine(
        embeddings_manager=embeddings_manager,
        llm_manager=llm_manager,
        retrieval_method="vector",
        top_k=3
    )
    
    # Load a document
    document_path = "data/sample_contract.pdf"
    qa_engine.load_document(document_path)
    
    # Ask a question
    question = "What are the payment terms in this contract?"
    result = qa_engine.answer_question(question)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Based on context: {result['context'][:200]}...")
    
    # Generate a summary
    summary = qa_engine.generate_document_summary(max_words=100)
    print(f"\nDocument Summary (100 words):\n{summary}")