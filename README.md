```markdown
# Document RAG Streamlit App

A Streamlit application that lets users upload PDF, TXT, or Markdown documents and ask questions about their content.

The app uses Google Gemini to generate embeddings and answers, while FAISS performs similarity searches over document chunks.

## Features

- Upload PDF, TXT, and Markdown files
- Split documents into searchable text chunks
- Generate embeddings using Gemini
- Retrieve relevant content using FAISS
- Ask questions through a Streamlit chat interface
- Cache the document index for faster responses

## Requirements

- Python 3.10+
- Google Gemini API key

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_google_api_key
```

Do not upload the `.env` file to GitHub.

## Run the application

```bash
streamlit run rag_streamlit.py
```

Open the URL shown in the terminal, upload a document, and start asking questions.

## Project Structure

```text
.
├── rag_streamlit.py
├── rag-model/
│   └── rag.py
└── requirements.txt
```

## How it works

1. The uploaded document is loaded and divided into text chunks.
2. Each chunk is converted into a numerical embedding using Gemini.
3. The embeddings are stored in a FAISS vector index.
4. When a question is asked, the most relevant chunks are retrieved.
5. Gemini generates an answer using only the retrieved document content.

## Supported file types

- PDF
- TXT
- Markdown

## License

This project is for educational and demonstration purposes.
```
