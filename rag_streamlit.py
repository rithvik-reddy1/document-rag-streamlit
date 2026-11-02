"""Streamlit web interface for the console RAG application."""

from __future__ import annotations

import hashlib
import ast
import importlib.util
import logging
import os
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
for logger_name in ("pypdf", "fontTools", "langchain_community"):
	logging.getLogger(logger_name).setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


st.set_page_config(
	page_title="Document desk",
	page_icon=":material/menu_book:",
	layout="centered",
)


def load_rag_module():
	"""Load the existing RAG functions from the console implementation."""
	module_path = Path(__file__).parent / "rag-model" / "rag.py"
	spec = importlib.util.spec_from_file_location("rag_core", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load RAG core from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


rag = load_rag_module()
load_dotenv()


def get_google_api_key() -> str | None:
	"""Read the Gemini key from the environment or Streamlit secrets."""
	api_key = os.getenv("GOOGLE_API_KEY")
	if api_key:
		return api_key

	try:
		return st.secrets.get("GOOGLE_API_KEY")
	except FileNotFoundError:
		return None


@st.cache_resource(show_spinner=False)
def build_index(file_bytes: bytes, file_name: str):
	"""Create and cache a FAISS index for the uploaded document."""
	suffix = Path(file_name).suffix.lower()
	with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
		temporary_file.write(file_bytes)
		temporary_path = Path(temporary_file.name)

	try:
		documents = rag.load_document(temporary_path)
		return rag.build_vector_store(documents)
	finally:
		temporary_path.unlink(missing_ok=True)


@st.cache_resource(show_spinner=False)
def get_chat_model(model_name: str, api_key: str) -> ChatGoogleGenerativeAI:
	return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

 
def normalize_answer(answer: object) -> str:
	"""Extract displayable text from Gemini's structured response content."""
	structured_answer = answer
	if isinstance(answer, str):
		try:
			structured_answer = ast.literal_eval(answer)
		except (SyntaxError, ValueError):
			return answer

	text_parts: list[str] = []

	def collect_text(value: object) -> None:
		if isinstance(value, dict):
			text = value.get("text")
			if isinstance(text, str):
				text_parts.append(text)
			elif text is not None:
				collect_text(text)
		elif isinstance(value, list):
			for item in value:
				collect_text(item)

	collect_text(structured_answer)
	return "\n".join(text_parts).strip() or str(answer)


def reset_chat() -> None:
	st.session_state.messages = []


st.session_state.setdefault("messages", [])
st.session_state.setdefault("document_id", None)

with st.sidebar:
	st.header("Document desk", icon=":material/menu_book:")
	st.caption("Upload a document, then ask questions grounded in its content.")
	uploaded_file = st.file_uploader(
		"Choose a document",
		type=["pdf", "txt", "md"],
		help="Supported formats: PDF, TXT, and Markdown.",
	)
	model_name = st.text_input(
		"Gemini model",
		value=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
	)
	st.button(
		"Clear conversation",
		icon=":material/delete_sweep:",
		on_click=reset_chat,
		width="stretch",
	)

st.title("Ask your document", anchor=False)
st.write("A focused workspace for finding answers inside your own files.")

if uploaded_file is None:
	with st.container(border=True):
		st.subheader("Start with a document", anchor=False)
		st.caption("Use the uploader in the sidebar to load a PDF, text file, or Markdown file.")
	st.stop()

file_bytes = uploaded_file.getvalue()
document_id = hashlib.sha256(file_bytes).hexdigest()
if st.session_state.document_id != document_id:
	st.session_state.document_id = document_id
	st.session_state.messages = []

api_key = get_google_api_key()
if not api_key:
	st.error("Add GOOGLE_API_KEY to your .env file or Streamlit secrets before continuing.")
	st.stop()

try:
	with st.status("Preparing your document", expanded=False) as status:
		vector_store = build_index(file_bytes, uploaded_file.name)
		model = get_chat_model(model_name, api_key)
		status.update(label="Document ready", state="complete")
except Exception as error:
	st.error(f"The document could not be prepared: {error}")
	st.stop()

st.caption(f"Loaded: {uploaded_file.name}")

for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

question = st.chat_input("Ask a question about your document")
if question:
	st.session_state.messages.append({"role": "user", "content": question})
	with st.chat_message("user"):
		st.markdown(question)

	with st.chat_message("assistant"):
		with st.spinner("Searching the document..."):
			try:
				answer = normalize_answer(
					rag.answer_question(question, vector_store, model)
				)
			except Exception as error:
				answer = f"I could not answer that right now: {error}"
			st.markdown(answer)

	st.session_state.messages.append({"role": "assistant", "content": answer})
