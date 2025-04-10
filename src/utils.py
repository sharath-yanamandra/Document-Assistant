"""
Utility functions for the Document Assistant.
"""
import os
import time
import hashlib
from typing import Dict, Any, List

def create_cache_key(file_path: str, params: Dict[str, Any] = None) -> str:
    """
    Create a cache key based on file path and optional parameters.
    
    Args:
        file_path: Path to the file
        params: Optional parameters to include in the key
        
    Returns:
        Cache key string
    """
    # Get file modification time
    mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
    
    # Create base key from file path and modification time
    key_parts = [file_path, str(mtime)]
    
    # Add params if provided
    if params:
        for k, v in sorted(params.items()):
            key_parts.append(f"{k}={v}")
    
    # Create hash
    key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
    return key

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension without the dot
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()[1:] if ext else ""

def format_time(seconds: float) -> str:
    """
    Format seconds into a readable time string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def word_count(text: str) -> int:
    """
    Count the number of words in a text.
    
    Args:
        text: Text to count words in
        
    Returns:
        Number of words
    """
    return len(text.split())

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def timer(func):
    """
    Decorator to time a function's execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Function {func.__name__} took {format_time(duration)} to execute.")
        return result
    return wrapper