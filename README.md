# RAG-Based Document Chatbot

Upload PDFs and ask questions. AI answers only from your documents using Retrieval Augmented Generation (RAG).

## Features

- Upload PDF documents
- Ask questions about uploaded content
- AI retrieves relevant context and answers accurately
- Vector database for efficient retrieval

## Tech Stack

- **LangChain**: RAG pipeline orchestration
- **FAISS**: Vector database for embeddings
- **HuggingFace**: Embeddings model
- **OpenAI**: LLM for generation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Add your OpenAI API key to `.env`

## Usage

```bash
python rag_chatbot.py
```

Upload PDFs and start asking questions!

## Skills Demonstrated

- Retrieval Augmented Generation (RAG)
- Vector databases (FAISS)
- Document processing
- LangChain framework
- Embeddings and semantic search
