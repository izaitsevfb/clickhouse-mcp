"""Vector search utilities for ClickHouse documentation."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .docs_search import get_project_root, get_package_root


def get_default_index_path() -> Path:
    """Get the default path to the FAISS index."""
    # First try package data directory
    package_path = get_package_root() / "index"
    if package_path.exists() and (package_path / "index.faiss").exists():
        return package_path
        
    # Fall back to project root (development mode)
    project_path = get_project_root() / "index"
    return project_path


def create_faiss_index(
    chunks: List[Dict[str, Any]], 
    output_path: str,
    embeddings,
) -> None:
    """Create a FAISS index from document chunks.
    
    Args:
        chunks: List of document chunks to embed and index.
        output_path: Path where to save the FAISS index.
        embeddings: Embedding model to use.
    """
    # Convert to langchain Documents
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk['content'],
            metadata=chunk['metadata']
        )
        documents.append(doc)
    
    print(f"Creating FAISS index with {len(documents)} documents...")
    
    # Create the FAISS index
    vector_store = FAISS.from_documents(documents, embeddings)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the FAISS index
    vector_store.save_local(output_path)
    print(f"FAISS index saved to {output_path}")


def load_faiss_index(
    index_path: Optional[str] = None,
    embeddings = None,
) -> FAISS:
    """Load a FAISS index.
    
    Args:
        index_path: Path to the FAISS index. If None, uses the default path.
        embeddings: Embedding model to use for queries.
        
    Returns:
        FAISS vector store.
    """
    if index_path is None:
        index_path = get_default_index_path()
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def vector_search(
    vector_store: FAISS,
    query: str,
    num_results: int = 5
) -> List[Document]:
    """Search the vector store for documents similar to the query.
    
    Args:
        vector_store: The FAISS vector store.
        query: Query text.
        num_results: Number of results to return.
        
    Returns:
        List of documents similar to the query.
    """
    return vector_store.similarity_search(query, k=num_results)