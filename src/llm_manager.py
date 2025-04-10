"""
LLM Manager module for handling interactions with local language models.
"""
import os
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLMManager:
    """Manages interactions with a local LLM from HuggingFace."""
    
    def __init__(self, 
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 device: str = "auto",
                 load_in_8bit: bool = True):
        """
        Initialize the LLM manager.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ('auto', 'cpu', 'cuda')
            load_in_8bit: Whether to load the model in 8-bit precision to save memory
        """
        self.model_name = model_name
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.load_in_8bit = load_in_8bit and self.device == "cuda"
        
        # Load tokenizer and model
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with optimizations
        if self.load_in_8bit:
            # 8-bit quantization for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16
            )
        else:
            # Regular loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                self.model.to(self.device)
        
        # Create a text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        print(f"Model loaded successfully.")
    
    def generate_response(self, prompt: str, 
                         max_tokens: int = 512, 
                         temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated text response
        """
        # Update generation parameters
        self.pipe.max_new_tokens = max_tokens
        self.pipe.temperature = temperature
        
        # Generate response
        response = self.pipe(prompt)[0]['generated_text']
        
        # Remove the input prompt from the response to get only the new content
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
            
        return response
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer to a question given some context.
        
        Args:
            question: The question to answer
            context: Context information to help answer the question
            
        Returns:
            Generated answer
        """
        # Create a prompt that instructs the model to answer based on the context
        prompt = f"""You are a helpful assistant that answers questions based only on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.generate_response(prompt)
    
    def generate_summary(self, text: str, max_words: int = 100) -> str:
        """
        Generate a summary of the provided text.
        
        Args:
            text: Text to summarize
            max_words: Maximum number of words for the summary
            
        Returns:
            Generated summary
        """
        # Create a prompt that instructs the model to generate a summary
        prompt = f"""Summarize the following document in about {max_words} words:

{text}

Summary:"""
        
        return self.generate_response(prompt)


# Example usage
if __name__ == "__main__":
    # Initialize the LLM manager
    llm_manager = LLMManager()
    
    # Test generating a response
    prompt = "What are the key elements of a good contract?"
    response = llm_manager.generate_response(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")