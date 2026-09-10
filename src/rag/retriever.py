"""
Semantic Vector and Lexical RAG Retriever for context notes.
Indexes context notes and ranks candidate explanations using hybrid retrieval.
"""

import math
import re
from typing import List, Dict, Tuple, Optional
from src.rag.context_store import ContextNote, load_context_notes


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


class VectorRetriever:
    """
    In-memory semantic vector and keyword retrieval engine.
    Calculates TF-IDF vector embeddings with cosine similarity and metadata scoring.
    """
    def __init__(self, notes: Optional[List[ContextNote]] = None):
        self.notes: List[ContextNote] = notes or load_context_notes()
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self._build_index()

    def _build_index(self):
        doc_count = len(self.notes)
        doc_freq = {}
        
        # Calculate DF
        for note in self.notes:
            tokens = set(tokenize(f"{note.note_id} {note.applies_to} {note.date_str} {note.text}"))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1
                
        # Calculate IDF
        for t, count in doc_freq.items():
            self.idf[t] = math.log((doc_count + 1) / (count + 1)) + 1.0

        # Calculate TF-IDF vectors for documents
        for note in self.notes:
            tokens = tokenize(f"{note.note_id} {note.applies_to} {note.date_str} {note.text}")
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            
            vec = {}
            norm_sq = 0.0
            for t, count in tf.items():
                val = count * self.idf.get(t, 1.0)
                vec[t] = val
                norm_sq += val * val
            
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            for t in vec:
                vec[t] /= norm
            self.doc_vectors.append(vec)

    def retrieve(
        self,
        route: str,
        week_of: str,
        query: str = "",
        top_k: int = 3
    ) -> List[Tuple[ContextNote, float]]:
        """
        Retrieve candidate notes ranked by hybrid semantic similarity + temporal and route alignment.
        """
        q_tokens = tokenize(f"{route} {week_of} {query}")
        q_tf = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
            
        q_vec = {}
        norm_sq = 0.0
        for t, count in q_tf.items():
            val = count * self.idf.get(t, 1.0)
            q_vec[t] = val
            norm_sq += val * val
        
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        for t in q_vec:
            q_vec[t] /= norm

        scored_notes = []
        for idx, note in enumerate(self.notes):
            # Cosine similarity
            doc_vec = self.doc_vectors[idx]
            sim = sum(q_vec.get(t, 0.0) * val for t, val in doc_vec.items())
            
            # Boost score for route and date alignment
            boost = 0.0
            if note.is_route_applicable(route):
                boost += 0.5
            if note.is_date_applicable(week_of):
                boost += 0.5
                
            total_score = sim + boost
            scored_notes.append((note, total_score))
            
        scored_notes.sort(key=lambda x: x[1], reverse=True)
        return scored_notes[:top_k]
