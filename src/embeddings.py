"""
Embedding utilities for text vectorization
"""
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingGenerator:
    """Generate embeddings using HuggingFace models"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name (str): Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def get_embeddings(self, texts):
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def get_query_embedding(self, query):
        """
        Generate embedding for a query string
        
        Args:
            query (str): Query text
            
        Returns:
            list: Embedding vector
        """
        return self.embeddings.embed_query(query)
    
    def normalize_embeddings(self, embeddings):
        """
        Normalize embedding vectors to unit length
        
        Args:
            embeddings (list): List of embedding vectors
            
        Returns:
            list: Normalized embedding vectors
        """
        return [emb/np.linalg.norm(emb) for emb in embeddings]
    
    # Required methods for direct use as an embedding model in vector stores
    
    def embed_documents(self, texts):
        """
        Generate embeddings for a list of texts.
        Required for compatibility with vector stores.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text):
        """
        Generate embedding for a query string.
        Required for compatibility with vector stores.
        
        Args:
            text (str): Query text
            
        Returns:
            list: Embedding vector
        """
        return self.embeddings.embed_query(text)