"""
Tests for the QA Engine module.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from src.qa_engine import QAEngine
from src.embeddings import EmbeddingsManager
from src.llm_manager import LLMManager

class TestQAEngine:
    """Test cases for QAEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mocks for dependencies
        self.mock_embeddings_manager = MagicMock(spec=EmbeddingsManager)
        self.mock_llm_manager = MagicMock(spec=LLMManager)
        
        # Set up return values for the mocks
        self.mock_embeddings_manager.search.return_value = [
            {"chunk": "This is chunk 1 about payments.", "score": 0.8, "index": 0},
            {"chunk": "This is chunk 2 about terms.", "score": 0.7, "index": 1}
        ]
        self.mock_embeddings_manager.chunks = [
            "This is chunk 1 about payments.",
            "This is chunk 2 about terms.",
            "This is chunk 3 about something else."
        ]
        
        self.mock_llm_manager.generate_answer.return_value = "This is the answer."
        self.mock_llm_manager.generate_summary.return_value = "This is the summary."
        
        # Create the QA engine with mocked dependencies
        self.qa_engine = QAEngine(
            embeddings_manager=self.mock_embeddings_manager,
            llm_manager=self.mock_llm_manager
        )
        
        # Set document text
        self.qa_engine.document_text = "This is the full document text for testing."
    
    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        qa_engine = QAEngine(
            embeddings_manager=self.mock_embeddings_manager,
            llm_manager=self.mock_llm_manager
        )
        
        assert qa_engine.retrieval_method == "vector"
        assert qa_engine.top_k == 3
        assert qa_engine.document_text == ""
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        qa_engine = QAEngine(
            embeddings_manager=self.mock_embeddings_manager,
            llm_manager=self.mock_llm_manager,
            retrieval_method="keyword",
            top_k=5
        )
        
        assert qa_engine.retrieval_method == "keyword"
        assert qa_engine.top_k == 5
    
    @patch('src.qa_engine.DocumentProcessor')
    def test_load_document(self, mock_processor_class):
        """Test loading a document."""
        # Setup mock document processor
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        mock_processor.process_document.return_value = {
            "full_text": "This is the full document text.",
            "chunks": ["Chunk 1", "Chunk 2"],
            "num_chunks": 2
        }
        
        # Test loading a document
        self.qa_engine.load_document("fake_path.pdf")
        
        # Check that the processor was called correctly
        mock_processor_class.assert_called_once()
        mock_processor.process_document.assert_called_once_with("fake_path.pdf")
        
        # Check that embeddings were processed
        self.mock_embeddings_manager.process_chunks.assert_called_once_with(["Chunk 1", "Chunk 2"])
        
        # Check that embeddings were saved
        self.mock_embeddings_manager.save.assert_called_once()
        
        # Check that the document text was stored
        assert self.qa_engine.document_text == "This is the full document text."
    
    def test_retrieve_chunks_vector(self):
        """Test retrieving chunks using vector similarity."""
        chunks = self.qa_engine._retrieve_chunks_vector("Test question")
        
        # Check that search was called correctly
        self.mock_embeddings_manager.search.assert_called_once_with("Test question", top_k=3)
        
        # Check the returned chunks
        assert chunks == ["This is chunk 1 about payments.", "This is chunk 2 about terms."]
    
    def test_retrieve_chunks_keyword(self):
        """Test retrieving chunks using keyword matching."""
        # Set retrieval method to keyword
        self.qa_engine.retrieval_method = "keyword"
        
        # Test with a question containing keywords
        chunks = self.qa_engine._retrieve_chunks_keyword("What are the payment terms?")
        
        # Check that we got some chunks back
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_retrieve_relevant_context_vector(self):
        """Test retrieving relevant context using vector method."""
        self.qa_engine.retrieval_method = "vector"
        context = self.qa_engine.retrieve_relevant_context("Test question")
        
        # Check that search was called
        self.mock_embeddings_manager.search.assert_called_once()
        
        # Check that we got a context string back
        assert isinstance(context, str)
        assert "chunk 1" in context
        assert "chunk 2" in context
    
    def test_retrieve_relevant_context_keyword(self):
        """Test retrieving relevant context using keyword method."""
        # Set method to keyword
        self.qa_engine.retrieval_method = "keyword"
        
        # Mock the keyword retrieval method
        self.qa_engine._retrieve_chunks_keyword = MagicMock(
            return_value=["Keyword chunk 1", "Keyword chunk 2"]
        )
        
        context = self.qa_engine.retrieve_relevant_context("Test question")
        
        # Check that the keyword method was called
        self.qa_engine._retrieve_chunks_keyword.assert_called_once_with("Test question")
        
        # Check the context
        assert "Keyword chunk 1" in context
        assert "Keyword chunk 2" in context
    
    def test_answer_question(self):
        """Test answering a question."""
        # Mock retrieve_relevant_context
        self.qa_engine.retrieve_relevant_context = MagicMock(
            return_value="This is the relevant context."
        )
        
        # Test answering a question
        result = self.qa_engine.answer_question("Test question")
        
        # Check that context was retrieved
        self.qa_engine.retrieve_relevant_context.assert_called_once_with("Test question")
        
        # Check that the LLM was used to generate an answer
        self.mock_llm_manager.generate_answer.assert_called_once_with(
            "Test question", "This is the relevant context."
        )
        
        # Check the result structure
        assert result["question"] == "Test question"
        assert result["answer"] == "This is the answer."
        assert result["context"] == "This is the relevant context."
    
    def test_generate_document_summary(self):
        """Test generating a document summary."""
        summary = self.qa_engine.generate_document_summary(max_words=100)
        
        # Check that the LLM was used to generate a summary
        self.mock_llm_manager.generate_summary.assert_called_once()
        
        # Check that max_words was passed correctly
        assert self.mock_llm_manager.generate_summary.call_args[1]["max_words"] == 100
        
        # Check the summary
        assert summary == "This is the summary."