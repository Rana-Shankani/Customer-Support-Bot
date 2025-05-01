"""
Vector store implementation for document retrieval
"""
import os
import config
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStore:
    """Vector store for document chunks"""
    
    def __init__(self, embedding_model=None):
        """
        Initialize vector store
        
        Args:
            embedding_model: The embedding model to use
        """
        if embedding_model is None:
            print(f"Initializing default HuggingFaceEmbeddings with model: {config.EMBEDDING_MODEL}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        else:
            print(f"Using provided embedding model: {type(embedding_model).__name__}")
            self.embedding_model = embedding_model
            
        # Verify the embedding model has the required methods
        required_methods = ['embed_documents', 'embed_query']
        for method in required_methods:
            if not hasattr(self.embedding_model, method):
                raise AttributeError(f"Embedding model is missing required method: {method}")
                
        self.vector_store = None
    
    def create_from_texts(self, texts, persist_directory=None, metadatas=None):
        """
        Create a vector store from a list of texts
        
        Args:
            texts (list): List of text strings
            persist_directory (str, optional): Directory to persist the vector store
            metadatas (list, optional): List of metadata dicts for each text
            
        Returns:
            VectorStore: self
        """
        if not texts:
            raise ValueError("No texts provided for creating vector store")
            
        print(f"Creating vector store from {len(texts)} texts")
        
        # Use config directory if not specified
        persist_directory = persist_directory or config.VECTOR_STORE_PATH
        
        # Create simple metadata if none provided
        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(texts))]
        
        # For debugging, check a sample text
        if texts:
            sample_idx = min(5, len(texts)-1)
            sample_text = texts[sample_idx]
            print(f"Sample text ({len(sample_text)} chars): {sample_text[:100]}...")
        
        try:
            # Ensure directory exists
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                print(f"Ensuring directory exists: {persist_directory}")
            
            # Create Chroma vector store
            kwargs = {
                "texts": texts,
                "embedding": self.embedding_model,
                "metadatas": metadatas,
            }
            
            # Only add persist_directory if provided
            if persist_directory:
                kwargs["persist_directory"] = persist_directory
                
            self.vector_store = Chroma.from_texts(**kwargs)
            
            # Call persist() without arguments if persist_directory was provided
            if persist_directory and hasattr(self.vector_store, "persist"):
                try:
                    self.vector_store.persist()
                    print(f"Vector store created and persisted to {persist_directory}")
                except Exception as e:
                    print(f"Note: Error calling persist(): {e}")
                    print("This is expected with newer Chroma versions where data is auto-persisted")
            else:
                print("Vector store created (in-memory, not persisted)")
                
            return self
        except Exception as e:
            print(f"Error creating vector store: {e}")
            # Try to provide more diagnostics
            try:
                # Test embedding with a simple text
                test_embedding = self.embedding_model.embed_documents(["Test document for embedding"])
                print(f"Test embedding shape: {len(test_embedding[0])}")
            except Exception as embed_error:
                print(f"Failed to create test embedding: {embed_error}")
            raise
    
    def get_retriever(self, search_kwargs=None):
        """
        Get a retriever from the vector store
        
        Args:
            search_kwargs (dict, optional): Search parameters
            
        Returns:
            Retriever: A retriever object
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_from_texts first.")
        
        default_search_kwargs = {"k": config.TOP_K_RESULTS}
        if search_kwargs:
            default_search_kwargs.update(search_kwargs)
            
        return self.vector_store.as_retriever(search_kwargs=default_search_kwargs)
    
    def similarity_search(self, query, k=None):
        """
        Search for similar documents
        
        Args:
            query (str): Query string
            k (int): Number of results to return
            
        Returns:
            list: List of documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_from_texts first.")
        
        k = k or config.TOP_K_RESULTS
        
        # Detect if it's a single word query (excluding common stop words)
        stop_words = ["the", "and", "or", "but", "a", "an", "in", "on", "at", "to", "for", "with", "by"]
        words = [w for w in query.lower().split() if w not in stop_words]
        
        if len(words) == 1 and len(words[0]) >= 3:
            # For single word queries, use a higher k value to ensure better matches
            # and then do additional filtering at the application level
            enhanced_k = min(k * 3, 15)  # Retrieve more results but cap at 15
            print(f"Single word query detected: '{query}'. Using enhanced retrieval (k={enhanced_k})")
            return self.vector_store.similarity_search(query, k=enhanced_k)
        
        return self.vector_store.similarity_search(query, k=k)
    
    def load(self, directory):
        """
        Load the vector store from disk
        
        Args:
            directory (str): Directory to load the vector store from
            
        Returns:
            VectorStore: self
        """
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")
            
        print(f"Loading vector store from {directory}")
        
        # Load the vector store
        try:
            self.vector_store = Chroma(
                persist_directory=directory,
                embedding_function=self.embedding_model
            )
            print("Vector store loaded successfully")
            return self
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise