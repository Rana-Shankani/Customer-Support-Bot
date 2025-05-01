"""
JSON document loader for loading knowledge base
"""
import json
import re
from typing import Dict, List, Any

class JSONLoader:
    """Load and process JSON data for the knowledge base"""
    
    def __init__(self, file_path: str):
        """
        Initialize the JSON loader
        
        Args:
            file_path (str): Path to the JSON file
        """
        self.file_path = file_path
    
    def load_json(self) -> List[Dict[str, Any]]:
        """
        Load data from JSON file
        
        Returns:
            List[Dict[str, Any]]: JSON data as a list of dictionaries
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Add common acknowledgments to ensure they're properly handled
            acknowledgments = [
                {"question": "Thank you", "answer": "You're welcome! Is there anything else I can help you with?"},
                {"question": "Thanks", "answer": "You're welcome! Is there anything else I can help you with?"},
                {"question": "OK", "answer": "Great! Is there anything else I can help you with?"},
                {"question": "Okay", "answer": "Great! Is there anything else I can help you with?"},
                {"question": "Got it", "answer": "Excellent! Let me know if you need anything else."}
            ]
            
            # Append these to the data
            data.extend(acknowledgments)
            
            return data
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {e}")
    
    def format_for_embedding(self) -> List[str]:
        """
        Format JSON data for embedding
        
        Returns:
            List[str]: List of formatted text strings
        """
        data = self.load_json()
        formatted_texts = []
        
        for item in data:
            if 'question' in item and 'answer' in item:
                # Create a context-rich format that includes both question and answer
                formatted_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                formatted_texts.append(formatted_text)
                
                # Also add variations of the question to improve matching
                question = item['question']
                
                # Remove question marks and convert to lowercase for better matching
                if question.endswith('?'):
                    alt_question = f"Question: {question[:-1]}\nAnswer: {item['answer']}"
                    formatted_texts.append(alt_question)
                    
                # Add a statement version (convert question to statement)
                if question.lower().startswith('how'):
                    # Add verb phrase version without the question words
                    verb_phrase = question[3:].strip().lower()
                    statement = f"Question: Information about {verb_phrase}\nAnswer: {item['answer']}"
                    formatted_texts.append(statement)
                    
                    # Add direct action version without "how do I"
                    # Example: "How do I redeem a gift card?" → "redeem a gift card"
                    action_phrase = re.sub(r'^how\s+do\s+i\s+', '', question.lower().rstrip('?'))
                    if action_phrase != question.lower().rstrip('?'):  # Only if it actually changed
                        formatted_texts.append(f"Question: {action_phrase}\nAnswer: {item['answer']}")
                        
                elif question.lower().startswith('what'):
                    statement = f"Question: Information about {question[4:].strip().lower()}\nAnswer: {item['answer']}"
                    formatted_texts.append(statement)
                    
                    # Add direct noun phrase without "what is" or "what are"
                    noun_phrase = re.sub(r'^what\s+(is|are)\s+', '', question.lower().rstrip('?'))
                    if noun_phrase != question.lower().rstrip('?'):  # Only if it actually changed
                        formatted_texts.append(f"Question: {noun_phrase}\nAnswer: {item['answer']}")
                        
                elif question.lower().startswith('where'):
                    statement = f"Question: Location for {question[5:].strip().lower()}\nAnswer: {item['answer']}"
                    formatted_texts.append(statement)
                elif question.lower().startswith('can'):
                    statement = f"Question: Information about whether {question[3:].strip().lower()}\nAnswer: {item['answer']}"
                    formatted_texts.append(statement)
                elif question.lower().startswith('do'):
                    statement = f"Question: Information about whether {question[2:].strip().lower()}\nAnswer: {item['answer']}"
                    formatted_texts.append(statement)

                # Add imperative form (command form) for how-to questions
                if question.lower().startswith('how'):
                    imperative_match = re.search(r'how\s+(?:do|can|to|would|should)\s+I\s+([a-z]+)', question.lower())
                    if imperative_match:
                        verb = imperative_match.group(1)
                        imperative = question.lower().replace(f"how do i {verb}", verb)
                        imperative = imperative.replace(f"how can i {verb}", verb)
                        imperative = imperative.replace(f"how to {verb}", verb)
                        imperative = imperative.rstrip('?')
                        formatted_texts.append(f"Question: {imperative}\nAnswer: {item['answer']}")
                    
                # Add keyword-based version
                keywords = [word.lower() for word in question.split() 
                           if len(word) > 3 and word.lower() not in 
                           ['what', 'where', 'when', 'how', 'who', 'why', 'can', 'does', 'do', 'is', 'are', 'will']]
                if keywords:
                    keyword_text = f"Question: {' '.join(keywords)}\nAnswer: {item['answer']}"
                    formatted_texts.append(keyword_text)
                
                # Add single-word keyword variations for better matching with single-word queries
                for keyword in keywords:
                    # Only add meaningful keywords (e.g., "refund", "password", "order")
                    if len(keyword) >= 4 and keyword not in ["about", "from", "with", "information", "have", "there"]:
                        single_keyword_text = f"Question: {keyword}\nAnswer: {item['answer']}"
                        formatted_texts.append(single_keyword_text)
                
                # Add topic-specific variations for important types of queries
                if "payment" in question.lower() or "pay" in question.lower():
                    payment_variations = [
                        f"Question: payment methods\nAnswer: {item['answer']}",
                        f"Question: payment options\nAnswer: {item['answer']}",
                        f"Question: how to pay\nAnswer: {item['answer']}",
                        f"Question: accepted payment methods\nAnswer: {item['answer']}",
                        f"Question: payment information\nAnswer: {item['answer']}",
                        f"Question: I want to know about payment\nAnswer: {item['answer']}"
                    ]
                    formatted_texts.extend(payment_variations)
                    
                # Add special handling for gift card redemption
                if "gift card" in question.lower() and ("redeem" in question.lower() or "use" in question.lower()):
                    gift_card_variations = [
                        f"Question: redeem a gift card\nAnswer: {item['answer']}",
                        f"Question: redeem gift card\nAnswer: {item['answer']}",
                        f"Question: using gift card\nAnswer: {item['answer']}",
                        f"Question: gift card redemption\nAnswer: {item['answer']}",
                        f"Question: gift card code\nAnswer: {item['answer']}",
                        f"Question: use my gift card\nAnswer: {item['answer']}"
                    ]
                    formatted_texts.extend(gift_card_variations)
                
                # Add variations for acknowledgment phrases
                if "thank you" in question.lower() or "thanks" in question.lower() or "ok" in question.lower().split() or "okay" in question.lower():
                    acknowledgment_variations = [
                        f"Question: tnx\nAnswer: {item['answer']}",
                        f"Question: thx\nAnswer: {item['answer']}",
                        f"Question: ty\nAnswer: {item['answer']}",
                        f"Question: k\nAnswer: {item['answer']}",
                        f"Question: great\nAnswer: {item['answer']}",
                        f"Question: good\nAnswer: {item['answer']}",
                        f"Question: got it\nAnswer: {item['answer']}",
                        f"Question: cool\nAnswer: {item['answer']}",
                        f"Question: fine\nAnswer: {item['answer']}",
                        f"Question: perfect\nAnswer: {item['answer']}"
                    ]
                    formatted_texts.extend(acknowledgment_variations)
        
        print(f"Created {len(formatted_texts)} formatted texts from {len(data)} FAQ items")
        return formatted_texts
    
    def get_formatted_dict(self) -> Dict[str, str]:
        """
        Get formatted data as a dictionary
        
        Returns:
            Dict[str, str]: Dictionary mapping questions to answers
        """
        data = self.load_json()
        qa_dict = {}
        
        for item in data:
            if 'question' in item and 'answer' in item:
                # Add the original question
                qa_dict[item['question']] = item['answer']
                
                # Also add common variations to the direct lookup dictionary
                question = item['question']
                
                # Add without question mark
                if question.endswith('?'):
                    qa_dict[question[:-1]] = item['answer']
                
                # Add direct action version for "how do I" questions
                if question.lower().startswith('how do i '):
                    action_version = question.lower().replace('how do i ', '').rstrip('?')
                    qa_dict[action_version] = item['answer']
                
                # Special handling for gift card redemption
                if "gift card" in question.lower() and ("redeem" in question.lower() or "use" in question.lower()):
                    qa_dict["redeem a gift card"] = item['answer']
                    qa_dict["redeem gift card"] = item['answer']
                    qa_dict["using gift card"] = item['answer']
                    qa_dict["gift card redemption"] = item['answer']
        
        return qa_dict