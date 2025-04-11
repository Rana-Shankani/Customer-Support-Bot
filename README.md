# RAG Customer Support Chatbot

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.22+-green.svg)
![FAISS](https://img.shields.io/badge/FAISS-1.10.0+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.0.1+-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready customer support chatbot that uses Retrieval Augmented Generation (RAG) to answer questions based on your product documentation.

![RAG Chatbot Demo](https://user-images.githubusercontent.com/your-username/your-repo/raw/main/docs/images/demo.gif)

## 🌟 Features

- 📄 **Document Processing**: Upload PDFs and text files
- ✂️ **Automatic Chunking**: Smart text splitting for better context retrieval
- 🔍 **Vector Search**: FAISS-powered semantic search
- 🤖 **AI-Powered Answers**: Generate precise answers using LLMs
- 🌐 **Web Interface**: Easy-to-use chat interface for non-technical users

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A HuggingFace API token

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-customer-support-chatbot.git
cd rag-customer-support-chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your HuggingFace API token
```

### Running the Application

```bash
python app.py
```

Then open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 📊 How It Works

![RAG Architecture](https://user-images.githubusercontent.com/your-username/your-repo/raw/main/docs/images/architecture.png)

1. **Upload Documents**: The system extracts text from your PDFs and text files
2. **Create Chunks**: Documents are split into manageable pieces
3. **Generate Embeddings**: Each chunk is converted into a vector embedding
4. **Build Index**: Embeddings are stored in a FAISS vector database
5. **Answer Questions**: The system retrieves relevant context and generates answers

## 🔧 Configuration

Adjust the following settings in your `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_LLM_MODEL=google/flan-t5-large
```

## 🧩 Project Structure

```
rag-customer-support/
├── app.py                  # Main application
├── requirements.txt        # Dependencies
├── src/                    # Source code
│   ├── document_loader.py  # Document processing
│   ├── embeddings.py       # Text embedding
│   ├── vector_store.py     # FAISS vector storage
│   └── qa_chain.py         # Question answering chain
├── utils/                  # Utilities
│   └── text_processing.py  # Text splitting functions
└── templates/              # Web interface
    └── index.html          # Main template
```

## 📚 Use Cases

- **Technical Support**: Answer customer questions about your product
- **Knowledge Base**: Make internal documentation searchable
- **Training**: Help new employees quickly find information
- **FAQ Automation**: Automatically handle common questions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the RAG framework
- FAISS for efficient vector similarity search
- HuggingFace for providing access to large language models
- The open-source AI community
