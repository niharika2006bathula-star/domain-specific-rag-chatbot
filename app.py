from __future__ import annotations

from typing import Any

import streamlit as st

from document_loader import load_documents_from_uploads
from rag_pipeline import answer_question
from vector_store import VectorStore, build_vector_store

MAX_FILE_SIZE_MB = 20


st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide",
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = False
if "document_summary" not in st.session_state:
    st.session_state.document_summary = {"processed_pdfs": 0, "processed_pages": 0, "chunks": 0, "errors": []}


def clear_chat() -> None:
    st.session_state.chat_history = []


def clear_documents() -> None:
    st.session_state.vector_store = None
    st.session_state.processed_documents = False
    st.session_state.document_summary = {"processed_pdfs": 0, "processed_pages": 0, "chunks": 0, "errors": []}


st.title("Domain-Specific RAG Chatbot")
st.caption("Upload PDFs and ask natural-language questions about the document content.")

with st.sidebar:
    st.header("PDF Upload")
    uploaded_files = st.file_uploader(
        "Choose one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write("Uploaded files:")
        for uploaded_file in uploaded_files:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.write(f"- {uploaded_file.name} ({size_mb:.2f} MB)")

    process_button = st.button("Process Documents", type="primary")
    clear_chat_button = st.button("Clear Chat")
    clear_documents_button = st.button("Clear Documents")

    if clear_chat_button:
        clear_chat()
    if clear_documents_button:
        clear_documents()

    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF before processing.")
        else:
            with st.spinner("Processing PDFs and building the document index..."):
                chunks, summary = load_documents_from_uploads(uploaded_files, max_size_mb=MAX_FILE_SIZE_MB)

                if chunks:
                    st.session_state.vector_store = build_vector_store(chunks)
                    st.session_state.processed_documents = True
                    st.session_state.document_summary = summary
                    st.success(
                        f"Processed {summary['processed_pdfs']} PDFs, {summary['processed_pages']} pages, and {summary['chunks']} chunks."
                    )
                else:
                    st.session_state.vector_store = None
                    st.session_state.processed_documents = False
                    st.session_state.document_summary = summary
                    st.warning("No readable document content was found. OCR may be required.")

                if summary.get("errors"):
                    for error in summary["errors"]:
                        st.warning(error)


st.subheader("Document Status")
if st.session_state.processed_documents and st.session_state.vector_store is not None:
    summary = st.session_state.document_summary
    st.metric("Processed PDFs", summary.get("processed_pdfs", 0))
    st.metric("Text pages", summary.get("processed_pages", 0))
    st.metric("Chunks", summary.get("chunks", 0))
else:
    st.info("Upload and process PDF files to enable question answering.")

if st.session_state.document_summary.get("errors"):
    st.write("Processing warnings:")
    for error in st.session_state.document_summary["errors"]:
        st.warning(error)

st.subheader("Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources:")
            for source in message["sources"]:
                st.caption(f"- {source['source']} (Page {source['page_number']})")

question = st.chat_input(
    "Ask a question about your uploaded PDFs",
    disabled=not st.session_state.processed_documents,
)

if question:
    if st.session_state.vector_store is None:
        st.warning("No documents have been processed yet. Please upload and process PDFs first.")
    else:
        with st.spinner("Retrieving context and generating an answer..."):
            result = answer_question(question, st.session_state.vector_store)

        answer_text = result.get("answer", "I could not find this information in the uploaded documents.")
        source_passages = result.get("source_passages", [])

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": [
                    {"source": passage["source"], "page_number": passage["page_number"]}
                    for passage in source_passages
                ],
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            st.markdown(answer_text)
            if source_passages:
                st.caption("Sources:")
                for passage in source_passages:
                    st.caption(f"- {passage['source']} (Page {passage['page_number']})")
            st.caption(result.get("warning", "Answers should be verified against the original documents."))

        st.rerun()
