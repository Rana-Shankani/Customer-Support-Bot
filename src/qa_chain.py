"""
Question answering chain for customer support
"""
import config
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class QAChain:
    """Question answering chain for customer support"""
    
    def __init__(self, retriever, model_id=None, support_contact=None, support_phone=None):
        """
        Initialize the QA chain
        
        Args:
            retriever: Document retriever
            model_id (str): HuggingFace model ID
            support_contact (str): Support contact email
            support_phone (str): Support contact phone
        """
        self.retriever = retriever
        self.model_id = model_id or config.LLM_MODEL
        self.support_contact = support_contact or config.SUPPORT_CONTACT
        self.support_phone = support_phone or config.SUPPORT_PHONE
        self.llm = self._init_llm()
        self.qa_chain = self._create_qa_chain()
    
    def _init_llm(self):
        """Initialize the language model pipeline"""
        gen_pipeline = pipeline(
            "text2text-generation",
            model=self.model_id,
            max_length=config.MAX_LENGTH,
            temperature=config.TEMPERATURE
        )
        return HuggingFacePipeline(pipeline=gen_pipeline)
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        # Create the template with direct variable substitution
        contact_info = f"{self.support_contact} or call {self.support_phone}"
        
        prompt_template = f"""Answer the question based only on the following context:

{{context}}

Question: {{question}}

If the question cannot be answered based on the context alone, or if the answer would be incomplete, state: "I don't have enough information about that in my knowledge base. Please contact our support team at {contact_info}."

Give a complete, detailed answer with all relevant information from the context. Answer:"""

        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def is_greeting(self, question):
        """Check if the question is a greeting"""
        q = question.strip().lower()
        return q in ["hi", "hello", "hey", "greetings"]
    
    def is_acknowledgment(self, question):
        """Check if the question is an acknowledgment or thanks"""
        q = question.strip().lower()
        acknowledgments = [
            "thank you", "thanks", "thx", "tnx", "thank", "thanks!", "thank you!",
            "ok", "okay", "k", "great", "good", "got it", "understood", "perfect",
            "cool", "sure", "alright", "all right", "fine", "nice", "np", "no problem",
            "welcome", "you're welcome", "youre welcome", "appreciate it", "ty"
        ]
        
        # Check for exact matches
        if q in acknowledgments:
            return True
            
        # Check for phrases that contain acknowledgments
        for phrase in acknowledgments:
            if phrase in q and len(q.split()) <= 5:  # Limit to short phrases
                return True
                
        return False
    
    def is_general_help_request(self, question):
        """Check if the question is a general help request"""
        # Remove punctuation and normalize whitespace
        q = question.strip().lower()
        q = q.rstrip("?!.,;:")
        
        # List of common help request phrases
        help_phrases = [
            "help", 
            "help me", 
            "i need help", 
            "can you help me", 
            "i have a problem", 
            "problem",
            "could you help me",
            "need assistance",
            "can you assist me",
            "support needed"
        ]
        
        # Check for exact matches first
        if q in help_phrases:
            return True
            
        # Check for phrases contained in the question
        for phrase in help_phrases:
            if phrase in q:
                return True
                
        return False
    
    def is_off_topic(self, question):
        """Check if the question is off-topic"""
        q = question.strip().lower()
        
        # List of common topics that are likely off-topic for customer support
        general_topics = [
            "food", "weather", "politics", "sports", "music", "history", 
            "geography", "water", "air", "animals", "travel", "finger", 
            "movie", "science", "math", "physics", "biology", "chemistry",
            "cook", "recipe", "game", "play", "exercise", "workout", "diet",
            "health", "medical", "doctor", "hospital", "school", "university",
            "college", "student", "teach", "learning", "book", "novel", "poem",
            "story", "author", "actor", "actress", "celebrity", "president",
            "country", "city", "river", "mountain", "ocean", "sea", "lake",
            "forest", "plant", "tree", "flower", "fruit", "vegetable"
        ]
        
        # Common customer support topics - these are NOT off-topic
        support_topics = [
            "account", "order", "payment", "shipping", "return", "refund",
            "product", "service", "website", "app", "subscription", "password",
            "email", "login", "sign", "contact", "support", "help", "issue",
            "problem", "cancel", "change", "update", "tracking", "delivery",
            "purchase", "buy", "checkout", "cart", "wishlist", "discount",
            "coupon", "promo", "code", "gift", "card", "warranty", "guarantee",
            "price", "cost", "fee", "charge", "bill", "invoice", "receipt",
            "member", "profile", "settings", "notification", "message"
        ]
        
        # If it contains support-related terms, probably not off-topic
        for topic in support_topics:
            if topic in q:
                return False
                
        # First check if it's a very short question containing general topics
        for topic in general_topics:
            if topic in q and len(q.split()) < 7:  # Expanded to slightly longer queries
                return True
                
        # Check for cooking-related questions specifically
        cooking_terms = ["cook", "recipe", "bake", "boil", "fry", "grill", "roast", "food", "meal", "pasta", "dish"]
        cooking_count = sum(1 for term in cooking_terms if term in q)
        if cooking_count >= 1:
            return True
            
        # More comprehensive check for questions that are clearly not product-related
        non_support_patterns = [
            "how do i make", "how to make", "how can i make", 
            "how do i cook", "how to cook", "how can i cook",
            "what is the best way to", "tell me about", "explain", 
            "who is", "where is", "when was", "why is", 
            "what are the symptoms", "how to treat", "how to cure"
        ]
        
        for pattern in non_support_patterns:
            if pattern in q:
                # Double-check it's not actually support-related
                support_related = any(topic in q for topic in support_topics)
                if not support_related:
                    return True
                
        return False
    
    def answer_question(self, question):
        """
        Answer a question using the QA chain
        
        Args:
            question (str): Question to answer
            
        Returns:
            str: Answer to the question
        """
        q = question.strip()
        
        # Handle greetings
        if self.is_greeting(q):
            return "Hello! How can I assist you today?"
            
        # Handle acknowledgments and thanks
        if self.is_acknowledgment(q):
            if "thank" in q.lower() or "thx" in q.lower() or "tnx" in q.lower() or "ty" in q.lower():
                return "You're welcome! Is there anything else I can help you with?"
            else:
                return "Great! Is there anything else I can help you with?"
        
        # Handle general help requests
        if self.is_general_help_request(q):
            return "I'd be happy to help. Could you please provide more details about your specific issue or question?"
        
        # Handle off-topic questions
        if self.is_off_topic(q):
            return (
                "I'm a customer support assistant focused on our products and services. "
                "For general knowledge questions like this, please use a search engine. "
                "Is there something about our products or services I can help you with?"
            )
        
        try:
            print(f"Processing question: '{q}'")
            
            # Check if it's a single-word query
            is_single_word = len(q.split()) == 1 and len(q) >= 3
            
            # Retrieve and answer
            result = self.qa_chain({"query": q})
            
            # Debug info
            print(f"Retrieved {len(result.get('source_documents', []))} source documents")
            for i, doc in enumerate(result.get('source_documents', [])[:2]):
                print(f"Doc {i+1} content: {doc.page_content[:100]}...")
            
            # Get the source documents and check relevance
            source_docs = result.get("source_documents", [])
            
            # Special handling for single-word queries
            if is_single_word:
                print(f"Single-word query detected: '{q}'")
                # For single-word queries, just check if the word appears in any source document
                if source_docs:
                    for doc in source_docs:
                        if q.lower() in doc.page_content.lower():
                            print(f"Found direct match for single word query '{q}' in document")
                            # Extract answer from matching document
                            if "answer:" in doc.page_content.lower():
                                parts = doc.page_content.lower().split("answer:", 1)
                                if len(parts) > 1:
                                    answer = parts[1].strip()
                                    # Ensure proper capitalization
                                    answer = answer[0].upper() + answer[1:]
                                    return answer
            
            # Regular relevance checking for multi-word queries
            # Check document relevance by looking for key terms
            query_terms = set([term.lower() for term in q.split() if len(term) > 3])
            is_relevant = False
            
            # Extra check for common terms that might appear in both on-topic and off-topic questions
            common_terms = {'make', 'find', 'what', 'when', 'where', 'how', 'best', 'good', 'need', 'want'}
            query_terms = query_terms - common_terms
            
            # For single-word queries, we're more lenient with relevance checking
            if is_single_word:
                is_relevant = True
            # Only consider it potentially relevant if we have meaningful terms
            elif query_terms:
                for doc in source_docs:
                    doc_terms = set([term.lower() for term in doc.page_content.split() if len(term) > 3])
                    overlap = query_terms.intersection(doc_terms)
                    # Some term overlap and not a general question
                    if len(overlap) >= 1 and len(query_terms) >= 2:
                        is_relevant = True
                        break
            
            # If we have source docs but none appear relevant to the query, it's likely off-topic
            if source_docs and not is_relevant and not self.is_off_topic(q):
                print("Documents don't appear relevant to the query, treating as off-topic")
                return (
                    "I don't have specific information about that topic in my knowledge base. "
                    "I'm focused on helping with questions about our products and services. "
                    "Is there something else I can assist you with?"
                )
            
            # Get the answer
            answer = result.get("result", "")
            print(f"Raw answer: {answer[:100]}...")
            
            # Prevent echo back
            if answer.lower().strip() == q.lower().strip():
                print("Answer matched question exactly - likely an echo, finding direct matches instead")
                # Try a direct lookup in the FAQ data
                for doc in source_docs:
                    if "answer" in doc.page_content.lower():
                        # Extract answer from the document content
                        parts = doc.page_content.split("Answer:", 1)
                        if len(parts) > 1:
                            return parts[1].strip()
                
            return answer
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return "I'm sorry, I encountered an error processing your question. Please try again or contact support."