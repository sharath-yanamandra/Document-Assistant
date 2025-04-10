"""
Tests for the embeddings module.
"""
import os
import pytest
import tempfile
import numpy as np
from src.embeddings import EmbeddingsManager

class TestEmbeddingsManager:
    """Test cases for EmbeddingsManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.embeddings_manager = EmbeddingsManager()
        self.test_chunks = [
            "This is the first test chunk.",
            "This is the second test chunk with more content.",
            "This third chunk contains information about payments and terms.",
            "The fourth chunk discusses contract termination policies."
        ]
    
    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        manager = EmbeddingsManager()
        assert manager.model_name == "all-MiniLM-L6-v2"
        assert manager.model is not None
        assert manager.index is None
        assert len(manager.chunks) == 0
    
    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        # Use another small model for testing
        manager = EmbeddingsManager(model_name="paraphrase-MiniLM-L3-v2")
        assert manager.model_name == "paraphrase-MiniLM-L3-v2"
        assert manager.model is not None
    
    def test_create_embeddings(self):
        """Test creating embeddings from text chunks."""
        embeddings = self.embeddings_manager.create_embeddings(self.test_chunks)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(self.test_chunks)
        assert embeddings.shape[1] > 0  # Should have some dimensions
    
    def test_process_chunks(self):
        """Test processing chunks by creating embeddings and building index."""
        self.embeddings_manager.process_chunks(self.test_chunks)
        
        assert self.embeddings_manager.chunks == self.test_chunks
        assert self.embeddings_manager.index is not None
        assert self.embeddings_manager.index.ntotal == len(self.test_chunks)
    
    def test_search(self):
        """Test searching for similar chunks."""
        self.embeddings_manager.process_chunks(self.test_chunks)
        
        # Search for payment-related content
        results = self.embeddings_manager.search("payment terms", top_k=2)
        
        assert len(results) <= 2
        assert isinstance(results, list)
        assert all("chunk" in result for result in results)
        assert all("score" in result for result in results)
        assert all("index" in result for result in results)
    
    def test_save_and_load(self):
        """Test saving and loading embeddings and index."""
        # Process chunks first
        self.embeddings_manager.process_chunks(self.test_chunks)
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the embeddings
            self.embeddings_manager.save(temp_dir)
            
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "chunks.pkl"))
            assert os.path.exists(os.path.join(temp_dir, "faiss_index.bin"))
            
            # Create a new manager and load the saved data
            new_manager = EmbeddingsManager()
            new_manager.load(temp_dir)
            
            # Verify the loaded data
            assert new_manager.chunks == self.test_chunks
            assert new_manager.index is not None
            assert new_manager.index.ntotal == len(self.test_chunks)
            
            # Test search with the loaded manager
            results = new_manager.search("payment terms", top_k=2)
            assert len(results) <= 2