"""
Tests for the document processor module.
"""
import os
import pytest
import tempfile
from src.document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        
        # Create a simple PDF for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_pdf_path = os.path.join(self.temp_dir.name, "test.pdf")
        
        # If we had a real PDF creation library, we would create a test PDF here
        # For now, we'll skip tests that require a real PDF
    
    def teardown_method(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        processor = DocumentProcessor()
        assert processor.chunk_strategy == "fixed"
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        processor = DocumentProcessor(
            chunk_strategy="semantic",
            chunk_size=300,
            chunk_overlap=30
        )
        assert processor.chunk_strategy == "semantic"
        assert processor.chunk_size == 300
        assert processor.chunk_overlap == 30
    
    def test_chunk_text_fixed_strategy(self):
        """Test text chunking with fixed strategy."""
        text = "This is a test document. " * 100  # Create a longer text
        processor = DocumentProcessor(chunk_strategy="fixed", chunk_size=100, chunk_overlap=0)
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 1
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_text_semantic_strategy(self):
        """Test text chunking with semantic strategy."""
        text = "This is a test document.\n\nThis is another paragraph.\n\nAnd a third one."
        processor = DocumentProcessor(chunk_strategy="semantic", chunk_size=100, chunk_overlap=0)
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 0
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.skipif(not os.path.exists("data/sample_contract.pdf"), 
                        reason="Sample PDF not available")
    def test_process_document(self):
        """Test processing a document (requires sample PDF)."""
        # This test will be skipped if the sample PDF doesn't exist
        processor = DocumentProcessor()
        result = processor.process_document("data/sample_contract.pdf")
        
        assert "full_text" in result
        assert "chunks" in result
        assert "num_chunks" in result
        assert isinstance(result["chunks"], list)
        assert result["num_chunks"] > 0