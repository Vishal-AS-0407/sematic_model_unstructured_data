"""
Utility functions for the semantic PDF model pipeline
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)

def save_json(data: Dict[Any, Any], filepath: str) -> None:
    """Save data to a JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    get_logger(__name__).info(f"Saved JSON to {filepath}")

def load_json(filepath: str) -> Dict[Any, Any]:
    """Load data from a JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    get_logger(__name__).info(f"Loaded JSON from {filepath}")
    return data

def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

        # Break if we're at the end
        if end >= text_length:
            break

    return chunks

def validate_api_key(api_key: str) -> bool:
    """Validate that API key is set"""
    if not api_key or api_key == "your_openai_api_key_here":
        get_logger(__name__).error("OpenAI API key not set. Please set OPENAI_API_KEY in .env file")
        return False
    return True
