"""
Document processor module for parsing and chunking PDF documents.
"""
import os
from typing import List, Dict, Any, Optional
import PyPDF2
#from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

class DocumentProcessor:
    """Processes PDF documents by extracting text and chunking."""
    
    def __init__(self, chunk_strategy: str = "fixed", 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_strategy: Strategy for chunking ('fixed' or 'semantic')
            chunk_size: Size of chunks in tokens/characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
                
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on the specified strategy.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if self.chunk_strategy == "fixed":
            # Fixed-size chunking (by tokens)
            text_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            # Semantic chunking (recursively by separator)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
        chunks = text_splitter.split_text(text)
        return chunks
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF document - extract text and create chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with full text and chunks
        """
        full_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(full_text)
        
        return {
            "full_text": full_text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "chunk_strategy": self.chunk_strategy
        }


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor(chunk_strategy="fixed", chunk_size=500)
    pdf_path = "data/sample_contract.pdf"  # Update with your PDF path
    
    try:
        result = processor.process_document(pdf_path)
        print(f"Document processed successfully.")
        print(f"Total chunks: {result['num_chunks']}")
        print(f"First chunk: {result['chunks'][0][:100]}...")
    except Exception as e:
        print(f"Error processing document: {str(e)}")