# RAG Customer Support Chatbot

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-orange.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Free_Models-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A customer support chatbot that uses Retrieval Augmented Generation (RAG) to answer q
uestions based on a JSON knowledge base.

## Features

- **JSON Knowledge Base**: Easy to update and maintain FAQ format
- **Automatic Chunking**: Smart text splitting for better context retrieval
- **Vector Search**: Fast semantic search of knowledge base
- **AI-Powered Answers**: Generate precise answers using free LLMs
- **Modern UI**: Responsive chat interface with typing indicators
- **Configuration**: Flexible configuration via environment variables
- **FastAPI Backend**: High-performance API for chat interactions

## Quick Start

### Prerequisites

- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/Rana-Shankani/Customer-Support-Bot.git
cd customer-support-bot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the FastAPI server
uvicorn app:app --reload
```

Then open your browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000)

## How It Works

1. **Load JSON Knowledge Base**: The system loads FAQ data from a JSON file
2. **Create Embeddings**: Each FAQ is converted into a vector embedding
3. **Build Vector Index**: Embeddings are stored in a vector database
4. **User Queries**: When a user asks a question, it's converted to an embedding
5. **Retrieve Similar FAQs**: The system finds the most similar FAQs
6. **Generate Response**: An LLM uses the retrieved FAQs to generate a helpful answer

## Configuration

All configuration is managed through environment variables or the `config.py` file:

| Variable | Description | Default |
|----------|-------------|---------|
| KNOWLEDGE_SOURCE | Path to the JSON knowledge base | faqs.json |
| EMBEDDING_MODEL | HuggingFace embedding model | all-MiniLM-L6-v2 |
| LLM_MODEL | HuggingFace LLM model | google/flan-t5-small |
| TEMPERATURE | Temperature for LLM generation | 0.1 |
| CHUNK_SIZE | Size of text chunks | 500 |
| CHUNK_OVERLAP | Overlap between chunks | 50 |
| TOP_K_RESULTS | Number of similar documents to retrieve | 3 |
| API_HOST | Host to bind the API server | 0.0.0.0 |
| API_PORT | Port for the API server | 8000 |
| SUPPORT_CONTACT | Support email address | support@example.com |
| SUPPORT_PHONE | Support phone number | 1-800-123-4567 |

## JSON Knowledge Base Format

The knowledge base is a JSON file containing an array of question-answer pairs:

```json
[
  {
    "question": "How do I reset my password?",
    "answer": "Go to Account Settings and click on 'Reset Password'."
  },
  {
    "question": "What payment methods do you accept?",
    "answer": "We accept Visa, MasterCard, PayPal, and Apple Pay."
  }
]
```

## Project Structure

```
customer-support-bot/
├── app.py                  # FastAPI application
├── config.py               # Configuration settings
├── requirements.txt        # Dependencies
├── faqs.json               # JSON knowledge base
├── src/                    # Source code
│   ├── embeddings.py       # Text embedding
│   ├── vector_store.py     # Vector storage
│   └── qa_chain.py         # Question answering chain
├── utils/                  # Utilities
│   ├── json_loader.py      # JSON document loader
│   └── text_processing.py  # Text splitting functions
├── static/                 # Static files
│   └── css/                # CSS styles
│       └── styles.css      # Custom styles
└── templates/              # Web interface
    └── index.html          # Chat UI template
```

## Use Cases

- **Customer Support**: Answer common customer questions automatically
- **Internal Knowledge Base**: Make company FAQs searchable
- **Training**: Help new employees find information quickly
- **Documentation**: Make product documentation accessible through chat

## Customization

### Adding New FAQs

Simply add new entries to the `faqs.json` file:

```json
{
  "question": "Your new question here?",
  "answer": "The detailed answer to the question."
}
```

### Using a Different Embedding Model

Change the `EMBEDDING_MODEL` in `config.py` to use a different HuggingFace embedding model.

### Using a Different LLM

Change the `LLM_MODEL` in `config.py` to use a different HuggingFace model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG framework
- HuggingFace for providing access to large language models
- The open-source AI community
