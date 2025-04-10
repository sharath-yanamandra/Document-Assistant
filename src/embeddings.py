"""
Embeddings module for creating and managing vector representations of text.
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingsManager:
    """Manages the creation and retrieval of embeddings for document chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            NumPy array of embeddings
        """
        self.chunks = chunks
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Build a FAISS index for fast similarity search.
        
        Args:
            embeddings: NumPy array of embeddings
        """
        vector_dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index.add(embeddings.astype('float32'))
    
    def process_chunks(self, chunks: List[str]) -> None:
        """
        Process chunks by creating embeddings and building an index.
        
        Args:
            chunks: List of text chunks
        """
        embeddings = self.create_embeddings(chunks)
        self.build_index(embeddings)
    
    def save(self, save_dir: str) -> None:
        """
        Save the embeddings, index, and chunks to disk.
        
        Args:
            save_dir: Directory to save the files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save chunks
        with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save the index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.bin"))
    
    def load(self, save_dir: str) -> None:
        """
        Load embeddings, index, and chunks from disk.
        
        Args:
            save_dir: Directory containing the saved files
        """
        # Load chunks
        with open(os.path.join(save_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        
        # Load the index
        self.index = faiss.read_index(os.path.join(save_dir, "faiss_index.bin"))
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for the most similar chunks to a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with chunk text and similarity score
        """
        if self.index is None:
            raise ValueError("Index not built. Call process_chunks first.")
        
        # Create embedding for the query
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Ensure index is valid
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(distances[0][i]),  # Convert to float for JSON serialization
                    "index": int(idx)
                })
        
        return results


# Example usage
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    
    # Process a document
    processor = DocumentProcessor()
    pdf_path = "data/sample_contract.pdf"  # Update with your PDF path
    result = processor.process_document(pdf_path)
    
    # Create embeddings
    embeddings_manager = EmbeddingsManager()
    embeddings_manager.process_chunks(result["chunks"])
    
    # Test search
    query = "What are the payment terms?"
    search_results = embeddings_manager.search(query, top_k=3)
    
    print(f"Query: {query}")
    for i, result in enumerate(search_results):
        print(f"Result {i+1} (Score: {result['score']:.4f}):")
        print(f"{result['chunk'][:200]}...")
        print()