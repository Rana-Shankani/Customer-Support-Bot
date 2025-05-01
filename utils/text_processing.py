"""
Text processing utilities for chunking and preparing documents for embedding.
"""
from typing import Dict, List
import config

from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(documents: Dict[str, str], 
                  chunk_size: int = None, 
                  chunk_overlap: int = None) -> List[Dict]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents: Dictionary of document names to their text content
        chunk_size: The target size of each text chunk (in characters)
        chunk_overlap: The overlap between chunks to maintain context
        
    Returns:
        List of dictionaries, each with text content, source document,
        and optional embedding
    """
    # Use config values as defaults if not specified
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    # Define text splitter with parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Create a list to hold all text chunks with metadata
    document_chunks = []
    
    for doc_name, text in documents.items():
        chunks = text_splitter.split_text(text)
        # Store each chunk along with its document name metadata
        for chunk in chunks:
            document_chunks.append({
                "text": chunk, 
                "source": doc_name
            })
    
    return document_chunks

def split_text(text, chunk_size=None, chunk_overlap=None):
    """
    Split text into chunks with overlap using RecursiveCharacterTextSplitter
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    # Use config values as defaults if not specified
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    print(f"Splitting text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    
    if chunks:
        print(f"Split into {len(chunks)} chunks. First chunk size: {len(chunks[0])}")
    
    return chunks