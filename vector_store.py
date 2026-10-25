from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from document_loader import DocumentChunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "vector_store", "faiss_index.index")
FAISS_METADATA_PATH = os.path.join(os.path.dirname(__file__), "vector_store", "faiss_metadata.json")


@dataclass
class RetrievalResult:
    text: str
    source: str
    page_number: int
    score: float


class VectorStore:
    def __init__(self, chunks: list[DocumentChunk] | None = None):
        self.model = self._load_model()
        self.chunks: list[DocumentChunk] = []
        self.index: faiss.Index | None = None
        self.embedding_dim = 384
        if chunks:
            self.add_chunks(chunks)

    def _load_model(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(MODEL_NAME, device="cpu")
        except Exception as exc:  # noqa: BLE001 - surface clearer app-level error
            raise RuntimeError(
                "Failed to load the embedding model. Check your internet connection and ensure the model is available."
            ) from exc

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype="float32")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            self.chunks = []
            self.index = None
            return

        self.chunks = list(chunks)
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self._encode_texts(texts)
        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievalResult]:
        if not question or not str(question).strip():
            return []
        if self.index is None or not self.chunks:
            return []

        question_embedding = self._encode_texts([str(question).strip()])
        limit = min(top_k, len(self.chunks))
        scores, indices = self.index.search(question_embedding, limit)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievalResult(
                    text=chunk.text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    score=float(score),
                )
            )
        return results

    def save(self, index_path: str = FAISS_INDEX_PATH, metadata_path: str = FAISS_METADATA_PATH) -> None:
        if self.index is None:
            raise ValueError("Cannot save an empty vector store.")

        safe_dir = os.path.abspath(os.path.dirname(__file__))
        safe_index_path = os.path.abspath(index_path)
        safe_metadata_path = os.path.abspath(metadata_path)

        if not safe_index_path.startswith(safe_dir) or not safe_metadata_path.startswith(safe_dir):
            raise ValueError("Unsafe index save location detected.")

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

        metadata = {
            "chunks": [
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in self.chunks
            ],
            "embedding_dim": self.embedding_dim,
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False)

    def load(self, index_path: str = FAISS_INDEX_PATH, metadata_path: str = FAISS_METADATA_PATH) -> "VectorStore":
        safe_dir = os.path.abspath(os.path.dirname(__file__))
        safe_index_path = os.path.abspath(index_path)
        safe_metadata_path = os.path.abspath(metadata_path)

        if not safe_index_path.startswith(safe_dir) or not safe_metadata_path.startswith(safe_dir):
            raise ValueError("Unsafe index load location detected.")
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return VectorStore([])

        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        loaded_chunks = [
            DocumentChunk(
                text=item["text"],
                source=item["source"],
                page_number=int(item["page_number"]),
                chunk_index=int(item.get("chunk_index", 0)),
            )
            for item in metadata.get("chunks", [])
        ]

        store = VectorStore([])
        store.chunks = loaded_chunks
        store.embedding_dim = int(metadata.get("embedding_dim", 384))
        store.index = index
        return store


def build_vector_store(chunks: list[DocumentChunk]) -> VectorStore:
    return VectorStore(chunks)


def save_vector_store(store: VectorStore) -> None:
    store.save()


def load_vector_store() -> VectorStore:
    return VectorStore([]).load()
