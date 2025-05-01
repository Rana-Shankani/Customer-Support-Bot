import os
import sys
from typing import Dict, Any
import time
import re
from difflib import SequenceMatcher

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, BackgroundTasks
from pydantic import BaseModel

# Import config
import config

# Import components
from utils.json_loader import JSONLoader
from utils.text_processing import split_text
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.qa_chain import QAChain

# Initialize FastAPI app
app = FastAPI(
    title="Customer Support Bot",
    description="RAG-based Customer Support Bot API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Global variables
qa_system = None
model_loading = False
answer_cache = {}  # Simple in-memory cache for answers

# List of common acknowledgment phrases
ACKNOWLEDGMENTS = {
    "thank you": "You're welcome! Is there anything else I can help you with?",
    "thanks": "You're welcome! Is there anything else I can help you with?",
    "ok": "Great! Is there anything else I can help you with?",
    "okay": "Great! Is there anything else I can help you with?",
    "got it": "Excellent! Let me know if you need anything else.",
    "great": "I'm glad that helps! Feel free to ask if you have other questions."
}

# Load FAQs for exact match lookup
try:
    json_loader = JSONLoader(config.KNOWLEDGE_PATH)
    # Use the enhanced dictionary that includes variations
    faq_map = json_loader.get_formatted_dict()
    
    # Also add lowercase versions for better matching
    lowercase_faq_map = {q.lower().rstrip('?').strip(): a for q, a in faq_map.items()}
    faq_map.update(lowercase_faq_map)
    
    print(f"Loaded {len(faq_map)} FAQ entries (including variations)")
except Exception as e:
    faq_map = {}
    print(f"Error loading FAQs for exact lookup: {e}")

class ChatInput(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

def init_qa_system():
    """
    Initialize the QA system with the knowledge base
    
    Returns:
        QAChain: The QA chain
    """
    global qa_system, model_loading
    
    if model_loading:
        print("Model loading already in progress, waiting...")
        attempts = 0
        while model_loading and attempts < 10:
            time.sleep(1)
            attempts += 1
        if qa_system is not None:
            return qa_system
    
    # Check if QA system is already initialized
    if qa_system is not None:
        return qa_system
    
    try:
        model_loading = True
        print("Starting QA system initialization...")
        start_time = time.time()
        
        # Check if JSON knowledge base exists
        if not os.path.exists(config.KNOWLEDGE_PATH):
            model_loading = False
            raise FileNotFoundError(f"Knowledge base not found at {config.KNOWLEDGE_PATH}")
        
        # Check if we can load from existing vector store
        vector_store = VectorStore()
        if os.path.exists(config.VECTOR_STORE_PATH):
            print(f"Loading existing vector store from {config.VECTOR_STORE_PATH}")
            try:
                vector_store.load(config.VECTOR_STORE_PATH)
                retriever = vector_store.get_retriever(
                    search_kwargs={"k": config.TOP_K_RESULTS}
                )
                qa_system = QAChain(
                    retriever, 
                    model_id=config.LLM_MODEL,
                    support_contact=config.SUPPORT_CONTACT,
                    support_phone=config.SUPPORT_PHONE
                )
                model_loading = False
                duration = time.time() - start_time
                print(f"QA system loaded in {duration:.2f} seconds")
                return qa_system
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Will rebuild vector store from JSON")
                # Continue to rebuild
        
        # Load JSON and prepare knowledge base
        print(f"Building vector store from {config.KNOWLEDGE_PATH}")
        loader = JSONLoader(config.KNOWLEDGE_PATH)
        texts = loader.format_for_embedding()
        
        if not texts:
            model_loading = False
            raise ValueError("No valid question-answer pairs found in the knowledge base")
        
        # Split text into chunks if needed
        if config.CHUNK_SIZE > 0:
            chunks = []
            for text in texts:
                chunks.extend(split_text(
                    text, 
                    chunk_size=config.CHUNK_SIZE, 
                    chunk_overlap=config.CHUNK_OVERLAP
                ))
            print(f"JSON data split into {len(chunks)} chunks")
        else:
            chunks = texts
        
        # Create embeddings
        try:
            embedding_generator = EmbeddingGenerator(config.EMBEDDING_MODEL)
            
            # Test the embedding generator
            test_embedding = embedding_generator.embed_documents(["Test text"])
            print(f"Successfully created test embedding with shape: {len(test_embedding[0])}")
            
            # Ensure directory exists
            os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
            
            # Create vector store with persist_directory for auto-persistence
            vector_store = VectorStore(embedding_generator)
            vector_store.create_from_texts(
                texts=chunks,
                persist_directory=config.VECTOR_STORE_PATH
            )
            
            # Get retriever
            retriever = vector_store.get_retriever(
                search_kwargs={"k": config.TOP_K_RESULTS}
            )
            
            # Create QA chain
            qa_system = QAChain(
                retriever, 
                model_id=config.LLM_MODEL,
                support_contact=config.SUPPORT_CONTACT,
                support_phone=config.SUPPORT_PHONE
            )
            
            duration = time.time() - start_time
            print(f"QA system initialized in {duration:.2f} seconds")
            model_loading = False
            return qa_system
        except Exception as e:
            import traceback
            print(f"Error creating embeddings: {e}")
            print(traceback.format_exc())
            model_loading = False
            raise
            
    except Exception as e:
        print(f"Error initializing QA system: {e}")
        model_loading = False
        raise

def background_init_qa_system(background_tasks: BackgroundTasks):
    """Initialize QA system in the background"""
    try:
        background_tasks.add_task(init_qa_system)
        print("QA system initialization started in background")
    except Exception as e:
        print(f"Error starting background initialization: {e}")

@app.on_event("startup")
def startup_event():
    """Initialize QA system on startup"""
    try:
        # Start initialization in a background process to avoid blocking the server
        init_qa_system()
        print("QA system initialized successfully!")
    except Exception as e:
        print(f"Error initializing QA system: {e}")
        print("QA system will be initialized on first request")

@app.get("/", response_class=HTMLResponse)
def index(request: Request, background_tasks: BackgroundTasks):
    """Render the main page"""
    # Trigger background loading of models if needed
    global qa_system
    if qa_system is None:
        background_init_qa_system(background_tasks)
    return templates.TemplateResponse("index.html", {"request": request})

def is_acknowledgment(query: str) -> bool:
    """Check if the query is an acknowledgment"""
    query = query.lower().strip()
    
    # Direct matches
    if query in ACKNOWLEDGMENTS:
        return True
    
    # Partial matches
    for ack in ACKNOWLEDGMENTS:
        if ack in query and len(query.split()) <= 5:  # Limit to short phrases
            return True
            
    return False

@app.post("/api/chat", response_model=ChatResponse)
def chat(input_data: ChatInput) -> Dict[str, Any]:
    """Handle chat API requests"""
    question = input_data.question.strip()
    
    # Check cache first for quick responses
    cache_key = question.lower()
    if cache_key in answer_cache:
        print(f"Cache hit for question: '{question}'")
        return {"answer": answer_cache[cache_key]}
    
    # Quick response for acknowledgments
    if is_acknowledgment(question):
        print(f"Acknowledgment detected: '{question}'")
        for ack in ACKNOWLEDGMENTS:
            if ack in question.lower():
                response = ACKNOWLEDGMENTS[ack]
                answer_cache[cache_key] = response
                return {"answer": response}
        # Default response for other acknowledgments
        response = "Great! Is there anything else I can help you with?"
        answer_cache[cache_key] = response
        return {"answer": response}
    
    # Direct FAQ match lookup
    normalized_q = question.lower().rstrip('?').strip()
    if normalized_q in faq_map:
        response = faq_map[normalized_q]
        print(f"Exact FAQ match found: '{question}' -> '{response}'")
        answer_cache[cache_key] = response
        return {"answer": response}

    # Add substring-based fallback if only one FAQ contains the normalized query
    if len(normalized_q) > 3:
        matches = [key for key in faq_map if normalized_q in key]
        if len(matches) == 1:
            response = faq_map[matches[0]]
            print(f"Partial FAQ match found: '{matches[0]}' for question '{question}' -> '{response}'")
            answer_cache[cache_key] = response
            return {"answer": response}

    # Enhanced token-based fallback with improved matching
    tokens = [t.lower() for t in re.findall(r'\b\w{3,}\b', normalized_q)]
    if tokens:
        # First try requiring all tokens to match (existing logic)
        token_matches = [key for key in faq_map if all(token in key.lower().split() for token in tokens)]
        
        # If no exact token matches, try relaxed matching requiring most tokens to match
        if not token_matches and len(tokens) >= 2:
            # Calculate a match score for each FAQ
            scored_matches = []
            for key in faq_map:
                # Get significant tokens from the FAQ question
                faq_tokens = [t.lower() for t in re.findall(r'\b\w{3,}\b', key)]
                
                # Calculate token overlap ratio
                matching_tokens = sum(1 for token in tokens if any(token in faq_token for faq_token in faq_tokens))
                if matching_tokens > 0:
                    overlap_ratio = matching_tokens / len(tokens)
                    
                    # Calculate sequence similarity for additional precision
                    seq_similarity = SequenceMatcher(None, normalized_q, key.lower()).ratio()
                    
                    # Combined score (weighted to favor token matching)
                    combined_score = (overlap_ratio * 0.7) + (seq_similarity * 0.3)
                    
                    if combined_score > 0.5:  # Threshold for good match
                        scored_matches.append((key, combined_score))
            
            # Sort by score
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Use the best match if it's significantly better than the second-best
            if scored_matches and (len(scored_matches) == 1 or 
                                 (len(scored_matches) > 1 and 
                                  scored_matches[0][1] - scored_matches[1][1] > 0.2)):
                response = faq_map[scored_matches[0][0]]
                print(f"Enhanced token match found: '{scored_matches[0][0]}' (score: {scored_matches[0][1]:.2f}) for question '{question}' -> '{response}'")
                answer_cache[cache_key] = response
                return {"answer": response}
        
        # Use exact token match if available
        if token_matches:
            response = faq_map[token_matches[0]]
            print(f"Token FAQ match found: '{token_matches[0]}' for question '{question}' -> '{response}'")
            answer_cache[cache_key] = response
            return {"answer": response}

    # Initialize QA system if not already done
    global qa_system
    if qa_system is None:
        try:
            init_qa_system()
        except Exception as e:
            error_message = str(e)
            print(f"Error initializing QA system: {error_message}")
            # Provide a more user-friendly error message
            return {"answer": f"Sorry, I'm having trouble setting up my knowledge base. Please try again later or contact the administrator. Error: {error_message}"}
    
    # Fast path for common payment queries
    if any(term in question.lower() for term in ["payment", "pay", "paying", "paid", "payments"]):
        payment_answer = "We accept Visa, MasterCard, PayPal, and Apple Pay."
        # Store in cache
        answer_cache[cache_key] = payment_answer
        return {"answer": payment_answer}
    
    try:
        # Get answer from QA system
        answer = qa_system.answer_question(question)
        
        # Store in cache for future use
        answer_cache[cache_key] = answer
        
        return {"answer": answer}
    except Exception as e:
        error_message = str(e)
        print(f"Error generating answer: {error_message}")
        return {"answer": f"I apologize, but I encountered an issue while processing your request. Please try again or contact support. Error: {error_message}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host=config.API_HOST, 
        port=config.API_PORT, 
        reload=config.DEBUG
    )