from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
for logger_name in ("pypdf", "fontTools", "langchain_community"):
	logging.getLogger(logger_name).setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
	ChatGoogleGenerativeAI,
	GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}



def load_document(file_path: Path) -> list[Document]:
	"""Load a PDF, text, or Markdown document into LangChain documents."""
	if not file_path.is_file():
		raise FileNotFoundError(f"Document not found: {file_path}")

	suffix = file_path.suffix.lower()
	if suffix == ".pdf":
		return PyPDFLoader(str(file_path)).load()
	if suffix in {".txt", ".md"}:
		return [
			Document(
				page_content=file_path.read_text(encoding="utf-8"),
				metadata={"source": str(file_path)},
			)
		]

	supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
	raise ValueError(f"Unsupported document type '{suffix}'. Use: {supported}")


def build_vector_store(documents: Iterable[Document]) -> FAISS:
	"""Split the source document and index its chunks for similarity search."""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=1000,
		chunk_overlap=150,
		add_start_index=True,
	)
	chunks = splitter.split_documents(list(documents))
	if not chunks:
		raise ValueError("The document does not contain any text.")

	embeddings = GoogleGenerativeAIEmbeddings(
		model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
	)
	return FAISS.from_documents(chunks, embeddings)


def answer_question(
	question: str, vector_store: FAISS, model: ChatGoogleGenerativeAI
) -> str:
	"""Retrieve relevant chunks and answer using only their content."""
	documents = vector_store.similarity_search(question, k=4)
	context = "\n\n---\n\n".join(document.page_content for document in documents)
	prompt = ChatPromptTemplate.from_messages(
		[
			(
				"system",
				"Answer the question using only the provided context. "
				"If the answer is not in the context, say you could not find it "
				"in the document. Do not invent facts.\n\nContext:\n{context}",
			),
			("human", "{question}"),
		]
	)
	response = model.invoke(prompt.format_messages(context=context, question=question))
	return str(response.content)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Ask questions about a local PDF, text, or Markdown document."
	)
	parser.add_argument("document", type=Path, help="Path to the document")
	return parser.parse_args()


def main() -> None:
	load_dotenv()
	args = parse_args()

	if not os.getenv("GOOGLE_API_KEY"):
		raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings and chat.")

	documents = load_document(args.document)
	vector_store = build_vector_store(documents)
	model = ChatGoogleGenerativeAI(
		model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
	)

	print(f"Document ready: {args.document.name}")
	print("Ask questions about the document. Type 'exit' or press Ctrl+C to quit.")
	while True:
		try:
			question = input("\nYou: ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nGoodbye!")
			break

		if question.lower() in {"exit", "quit"}:
			print("Goodbye!")
			break
		if not question:
			continue

		try:
			print(f"\nAssistant: {answer_question(question, vector_store, model)}")
		except Exception as error:
			print(f"\nError: {error}")


if __name__ == "__main__":
	main()


