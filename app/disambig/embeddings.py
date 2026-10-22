"""Label/description embeddings for candidate retrieval and NE clustering.

GPTKB 2.0 runs Qwen3-Embedding-4B locally (sentence-transformers,
truncate_dim=1024) with a faiss HNSW index per element kind. This host has no
GPU and too little memory for that, so the same model is called through the
ScaDS.AI OpenAI-compatible /v1/embeddings endpoint instead, truncated to the
same 1024 dimensions and re-normalized. A miniGPTKB holds thousands, not
millions, of elements, so exact brute-force cosine search over an in-memory
numpy matrix replaces the approximate faiss index.
"""

from __future__ import annotations

import random
import sqlite3
import threading
import time
from typing import Callable

import numpy as np
from openai import OpenAI

EMBEDDING_DIM = 1024
_REQUEST_BATCH = 64


class EmbeddingClient:
    """Embeds texts, caching every vector in the run's database (texts recur
    constantly -- the same object label is looked up again on every mention)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        connection: sqlite3.Connection,
        stop_event: threading.Event,
        log: Callable[[str, str], None],
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
        self.model = model
        self.connection = connection
        self.stop_event = stop_event
        self.log = log

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length float32 vectors, one row per input text, in order."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        unique = list(dict.fromkeys(texts))
        vectors: dict[str, np.ndarray] = {}
        for start in range(0, len(unique), 500):
            chunk = unique[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            for text, blob in self.connection.execute(
                f"SELECT text, vec FROM text_embeddings WHERE text IN ({placeholders})", chunk
            ):
                vectors[text] = np.frombuffer(blob, dtype=np.float32)

        missing = [text for text in unique if text not in vectors]
        for start in range(0, len(missing), _REQUEST_BATCH):
            chunk = missing[start : start + _REQUEST_BATCH]
            fresh = self._request(chunk)
            self.connection.executemany(
                "INSERT OR REPLACE INTO text_embeddings (text, vec) VALUES (?, ?)",
                [(text, vec.tobytes()) for text, vec in zip(chunk, fresh)],
            )
            for text, vec in zip(chunk, fresh):
                vectors[text] = vec
        if missing:
            self.connection.commit()
        return np.stack([vectors[text] for text in texts]).astype(np.float32, copy=False)

    def _request(self, texts: list[str], max_retries: int = 6) -> list[np.ndarray]:
        attempt = 0
        while True:
            if self.stop_event.is_set():
                raise RuntimeError("Stopped while waiting for embeddings.")
            try:
                # An empty string is rejected by some servers; a single space
                # embeds as "nothing" just the same.
                response = self.client.embeddings.create(model=self.model, input=[t or " " for t in texts])
                rows = sorted(response.data, key=lambda item: item.index)
                out = []
                for row in rows:
                    vec = np.asarray(row.embedding, dtype=np.float32)[:EMBEDDING_DIM]
                    norm = float(np.linalg.norm(vec))
                    out.append(vec / norm if norm > 0 else vec)
                if len(out) != len(texts):
                    raise RuntimeError(f"Embedding endpoint returned {len(out)} vectors for {len(texts)} texts")
                return out
            except Exception as exc:
                attempt += 1
                if attempt > max_retries or self.stop_event.is_set():
                    raise
                delay = min(60.0, 2**attempt) + random.uniform(0, 1)
                self.log(
                    f"Embedding request failed ({type(exc).__name__}: {str(exc)[:160]}); retrying in {delay:.1f}s.",
                    "warning",
                )
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not self.stop_event.is_set():
                    time.sleep(0.25)


class VectorIndex:
    """Exact cosine (inner product of unit vectors) top-k search over the
    label embeddings of one element kind (entities, predicates or classes)."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self._matrix = np.zeros((1024, EMBEDDING_DIM), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        needed = len(self.ids) + len(ids)
        if needed > self._matrix.shape[0]:
            capacity = max(needed, self._matrix.shape[0] * 2)
            grown = np.zeros((capacity, EMBEDDING_DIM), dtype=np.float32)
            grown[: len(self.ids)] = self._matrix[: len(self.ids)]
            self._matrix = grown
        self._matrix[len(self.ids) : needed] = vectors
        self.ids.extend(ids)

    def search(self, queries: np.ndarray, k: int, min_similarity: float = 0.0) -> list[list[str]]:
        """search_scored() without the scores."""
        return [[element for element, _ in row] for row in self.search_scored(queries, k, min_similarity)]

    def search_scored(self, queries: np.ndarray, k: int, min_similarity: float = 0.0) -> list[list[tuple[str, float]]]:
        """For each query row, its k most similar distinct elements with their
        cosine similarity, best first, leaving out any below min_similarity.
        An element can have several rows (its label and its aliases); it ranks
        by its best-matching one."""
        n = len(self.ids)
        if n == 0 or len(queries) == 0:
            return [[] for _ in range(len(queries))]
        scores = queries @ self._matrix[:n].T
        result = []
        for row in range(len(queries)):
            width = min(n, k * 4)
            while True:
                if width < n:
                    candidates = np.argpartition(-scores[row], width - 1)[:width]
                else:
                    candidates = np.arange(n)
                ordered = candidates[np.argsort(-scores[row, candidates], kind="stable")]
                best: dict[str, float] = {}
                for i in ordered:
                    best.setdefault(self.ids[i], float(scores[row, i]))
                if len(best) >= k or width >= n:
                    break
                width = min(n, width * 4)
            result.append([(element, score) for element, score in list(best.items())[:k] if score >= min_similarity])
        return result


def cluster_labels(vectors: np.ndarray, distance_threshold: float = 0.15) -> list[int]:
    """Average-linkage agglomerative clustering on cosine distance, cut at
    `distance_threshold` -- the same clustering GPTKB 2.0 does with
    sklearn's AgglomerativeClustering(metric="cosine", linkage="average",
    distance_threshold=0.15) for its guarded NED parallelization."""
    if len(vectors) <= 1:
        return [0] * len(vectors)
    from scipy.cluster.hierarchy import fcluster, linkage

    # Unit vectors, so cosine distance is 1 - dot product; clip float error.
    linkage_matrix = linkage(vectors.astype(np.float64), method="average", metric="cosine")
    linkage_matrix[:, 2] = np.clip(linkage_matrix[:, 2], 0.0, None)
    # sklearn only merges below the threshold, fcluster at or below it.
    threshold = float(np.nextafter(distance_threshold, 0.0))
    return [int(label) for label in fcluster(linkage_matrix, t=threshold, criterion="distance")]
