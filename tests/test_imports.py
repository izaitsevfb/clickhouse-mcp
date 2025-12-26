"""
Test that all module imports work correctly.

This test ensures that all imports in the package are valid and
that dependencies are correctly specified, preventing runtime import
errors like ModuleNotFoundError.
"""

import unittest


class TestImports(unittest.TestCase):
    """Test that all critical imports work correctly."""

    def test_vector_search_imports(self):
        """Test that vector_search module imports correctly."""
        # This specifically tests that langchain imports are correct
        # (e.g., using langchain_core.documents instead of deprecated langchain.schema)
        from clickhouse_mcp.vector_search import (
            Document,
            create_faiss_index,
            load_faiss_index,
            vector_search,
            get_default_index_path,
        )

        # Verify Document class is usable
        doc = Document(page_content="test content", metadata={"key": "value"})
        self.assertEqual(doc.page_content, "test content")
        self.assertEqual(doc.metadata, {"key": "value"})

    def test_mcp_server_imports(self):
        """Test that mcp_server module imports correctly."""
        from clickhouse_mcp.mcp_server import mcp
        self.assertIsNotNone(mcp)

    def test_docs_search_imports(self):
        """Test that docs_search module imports correctly."""
        from clickhouse_mcp.docs_search import (
            load_chunks,
            get_project_root,
            get_package_root,
        )
        self.assertIsNotNone(load_chunks)

    def test_langchain_document_class(self):
        """Test that Document class from langchain_core works as expected."""
        from langchain_core.documents import Document

        # Test creating a document with content and metadata
        doc = Document(
            page_content="Sample documentation content",
            metadata={
                "source": "test.md",
                "title": "Test Document"
            }
        )

        self.assertEqual(doc.page_content, "Sample documentation content")
        self.assertEqual(doc.metadata["source"], "test.md")
        self.assertEqual(doc.metadata["title"], "Test Document")


if __name__ == "__main__":
    unittest.main()
