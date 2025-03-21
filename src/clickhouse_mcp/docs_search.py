"""ClickHouse documentation search utilities."""

import pickle
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


def get_package_root() -> Path:
    """Get the package root directory (where the module is installed)."""
    return Path(__file__).resolve().parent

def get_project_root() -> Path:
    """Get the project root directory."""
    # This function is kept for backward compatibility
    return get_package_root().parent.parent

def get_default_pickle_path() -> Path:
    """Get the default path to the pickle file containing document chunks."""
    # First try package data directory
    package_path = get_package_root() / "index" / "clickhouse_docs_chunks.pkl"
    if package_path.exists():
        return package_path
        
    # Fall back to project root (development mode)
    return get_project_root() / "index" / "clickhouse_docs_chunks.pkl"


def load_chunks(pickle_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load document chunks from a pickle file.
    
    Args:
        pickle_path: Path to the pickle file. If None, uses the default path.
        
    Returns:
        List of document chunks.
    """
    if pickle_path is None:
        pickle_path = get_default_pickle_path()
    
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def simple_search(chunks: List[Dict[str, Any]], query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Simple keyword search in document chunks.
    
    Args:
        chunks: List of document chunks to search in.
        query: Search query.
        num_results: Maximum number of results to return.
        
    Returns:
        List of document chunks matching the query, sorted by relevance.
    """
    query = query.lower()
    # Score each chunk based on the query (simple word matching)
    scored_chunks = []
    for chunk in chunks:
        content = chunk['content'].lower()
        score = 0
        
        # Check for exact phrase match
        if query in content:
            score += 10
        
        # Count individual words
        query_words = query.split()
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                score += content.count(word)
        
        # Also check metadata
        for k, v in chunk['metadata'].items():
            if isinstance(v, str) and query in v.lower():
                score += 5
            elif isinstance(v, str):
                for word in query_words:
                    if len(word) > 2 and word in v.lower():
                        score += 1
        
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Return top results
    return [chunk for _, chunk in scored_chunks[:num_results]]


def get_context_snippet(content: str, query: str, context_size: int = 50) -> str:
    """Extract a snippet of text around the query in the content.
    
    Args:
        content: The document content to extract from.
        query: The search query to find in the content.
        context_size: Number of characters to include before and after the match.
        
    Returns:
        A snippet of text around the query with ellipsis.
    """
    content_lower = content.lower()
    query_lower = query.lower()
    
    # First try to find the entire query
    query_pos = content_lower.find(query_lower)
    if query_pos != -1:
        start = max(0, query_pos - context_size)
        end = min(len(content), query_pos + len(query_lower) + context_size)
        return f"...{content[start:end]}..."
    
    # Try to find any of the query words
    query_words = [w for w in query_lower.split() if len(w) > 2]
    for word in query_words:
        query_pos = content_lower.find(word)
        if query_pos != -1:
            start = max(0, query_pos - context_size)
            end = min(len(content), query_pos + len(word) + context_size)
            return f"...{content[start:end]}..."
    
    # Just show the beginning of the content
    return f"{content[:100]}..."


def sample_random_chunks(chunks: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """Sample n random chunks from the collection.
    
    Args:
        chunks: List of document chunks to sample from.
        n: Number of chunks to sample.
        
    Returns:
        List of randomly sampled document chunks.
    """
    if n >= len(chunks):
        return chunks
    
    return random.sample(chunks, n)


def format_chunk_preview(chunk: Dict[str, Any], index: int, context_limit: int = 200) -> str:
    """Format a document chunk for display.
    
    Args:
        chunk: Document chunk to format.
        index: Index number for display.
        context_limit: Maximum number of characters to show from the content.
        
    Returns:
        Formatted string representation of the chunk.
    """
    result = []
    result.append(f"--- Chunk {index} ---")
    result.append(f"Document: {chunk['metadata']['document_title']}")
    result.append(f"Section: {chunk['metadata']['section_title']}")
    result.append(f"Source: {chunk['metadata']['path']}")
    result.append(f"Content Preview: {chunk['content'][:context_limit]}...")
    
    return "\n".join(result)