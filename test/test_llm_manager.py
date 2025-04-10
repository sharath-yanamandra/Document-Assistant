"""
Tests for the LLM Manager module.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.llm_manager import LLMManager

# Skip the actual model loading in tests to speed them up
@pytest.fixture
def mock_llm_manager():
    """Create a mocked LLM manager for testing."""
    with patch('src.llm_manager.AutoModelForCausalLM'), \
         patch('src.llm_manager.AutoTokenizer'), \
         patch('src.llm_manager.pipeline'):
        
        manager = LLMManager(device="cpu")
        
        # Mock the pipeline's call method
        manager.pipe = MagicMock()
        manager.pipe.return_value = [{'generated_text': 'Input prompt followed by generated response'}]
        
        yield manager

class TestLLMManager:
    """Test cases for LLMManager class."""
    
    def test_init_parameters(self):
        """Test initialization parameters."""
        with patch('src.llm_manager.LLMManager._load_model_and_tokenizer'):
            manager = LLMManager(
                model_name="different-model",
                device="cpu",
                load_in_8bit=False
            )
            
            assert manager.model_name == "different-model"
            assert manager.device == "cpu"
            assert manager.load_in_8bit is False
    
    def test_generate_response(self, mock_llm_manager):
        """Test generating a response."""
        # Setup the mock to return a specific output
        mock_llm_manager.pipe.return_value = [{'generated_text': 'Input prompt generated response'}]
        
        # Test with a simple prompt
        prompt = "Input prompt"
        response = mock_llm_manager.generate_response(prompt)
        
        # Check that the pipeline was called with the prompt
        mock_llm_manager.pipe.assert_called_once()
        assert mock_llm_manager.pipe.call_args[0][0] == prompt
        
        # Check the response
        assert response == " generated response"
    
    def test_generate_answer(self, mock_llm_manager):
        """Test generating an answer with context."""
        # Mock the generate_response method
        mock_llm_manager.generate_response = MagicMock(return_value="This is the answer.")
        
        # Test generating an answer
        question = "What is the payment schedule?"
        context = "Payments are due on the 1st of each month."
        answer = mock_llm_manager.generate_answer(question, context)
        
        # Check that generate_response was called with the expected prompt
        mock_llm_manager.generate_response.assert_called_once()
        prompt = mock_llm_manager.generate_response.call_args[0][0]
        
        # Verify the prompt contains both the question and context
        assert question in prompt
        assert context in prompt
        
        # Check the answer
        assert answer == "This is the answer."
    
    def test_generate_summary(self, mock_llm_manager):
        """Test generating a summary."""
        # Mock the generate_response method
        mock_llm_manager.generate_response = MagicMock(return_value="This is a summary.")
        
        # Test generating a summary
        text = "This is a long document that needs to be summarized."
        summary = mock_llm_manager.generate_summary(text, max_words=50)
        
        # Check that generate_response was called with the expected prompt
        mock_llm_manager.generate_response.assert_called_once()
        prompt = mock_llm_manager.generate_response.call_args[0][0]
        
        # Verify the prompt contains the text and word count
        assert text in prompt
        assert "50 words" in prompt
        
        # Check the summary
        assert summary == "This is a summary."