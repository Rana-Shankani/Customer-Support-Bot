"""
Configuration settings for the Customer Support Bot
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Knowledge base
KNOWLEDGE_SOURCE = os.environ.get('KNOWLEDGE_SOURCE', 'faqs.json')
KNOWLEDGE_PATH = os.path.join(BASE_DIR, KNOWLEDGE_SOURCE)

# Vector store settings
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')
VECTOR_STORE_TYPE = os.environ.get('VECTOR_STORE_TYPE', 'chroma')

# Embedding model settings
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# LLM model settings
LLM_MODEL = os.environ.get('LLM_MODEL', 'google/flan-t5-small')
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.1'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '512'))

# Text chunking settings
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '256'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '30'))

# Retrieval settings
TOP_K_RESULTS = int(os.environ.get('TOP_K_RESULTS', '2'))

# API settings
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', '8000'))
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Support contact
SUPPORT_CONTACT = os.environ.get('SUPPORT_CONTACT', 'support@example.com')
SUPPORT_PHONE = os.environ.get('SUPPORT_PHONE', '1-800-123-4567')

# Performance settings
CACHE_SIZE = int(os.environ.get('CACHE_SIZE', '100'))
MODEL_LOAD_TIMEOUT = int(os.environ.get('MODEL_LOAD_TIMEOUT', '30')) 