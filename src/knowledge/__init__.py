"""Knowledge module — vector retrieval and long-term memory via ChromaDB + Ollama."""
from src.knowledge.embedder import OllamaEmbedder
from src.knowledge.indexer import Indexer
from src.knowledge.search import Searcher, SearchResponse, SearchResult
from src.knowledge.store import KnowledgeStore

__all__ = [
    "KnowledgeStore",
    "OllamaEmbedder",
    "Indexer",
    "Searcher",
    "SearchResponse",
    "SearchResult",
]
