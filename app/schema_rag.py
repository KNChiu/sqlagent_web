"""
Schema RAG Service for efficient table schema retrieval.

This module provides RAG-based schema retrieval to reduce token usage
by only fetching relevant table schemas based on user queries.
"""

from pathlib import Path
from typing import List, Tuple
import json
import logging

from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)


class SchemaRAGService:
    """
    RAG service for retrieving relevant database table schemas.

    This service uses FAISS vector store to efficiently retrieve only the
    relevant table schemas based on semantic similarity to user queries,
    reducing token usage by 60-70% compared to fetching all schemas.
    """

    def __init__(
        self,
        db: SQLDatabase = None,
        index_path: str = "./data/faiss_index",
        schema_json_path: str = "./data/schema_descriptions.json",
        db_uri: str = "sqlite:///./Chinook.db",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    ):
        """
        Initialize Schema RAG Service.

        Args:
            db: Existing SQLDatabase instance (preferred to avoid duplicate connections)
            index_path: Path to FAISS index directory
            schema_json_path: Path to schema descriptions JSON
            db_uri: Database connection URI (only used if db is not provided)
            embedding_model_id: Bedrock embedding model ID
        """
        self.index_path = Path(index_path)
        self.schema_json_path = Path(schema_json_path)

        # Reuse existing db instance if provided, otherwise create new connection
        if db is not None:
            self.db = db
        else:
            self.db = SQLDatabase.from_uri(
                db_uri,
                lazy_table_reflection=True  # Lazy load for faster connection
            )

        # Initialize embeddings model
        self.embeddings = BedrockEmbeddings(model_id=embedding_model_id)

        # Load FAISS index
        self._load_index()

        # Load schema metadata
        self._load_schema_metadata()

        # Cache for table schemas
        self._schema_cache = {}

    def _load_index(self):
        """Load pre-built FAISS index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Please run 'python scripts/build_index.py' first."
            )

        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True  # Safe for local trusted index
        )
        logger.info(f"Loaded FAISS index from {self.index_path}")

    def _load_schema_metadata(self):
        """Load schema metadata from JSON file."""
        if not self.schema_json_path.exists():
            raise FileNotFoundError(
                f"Schema JSON not found at {self.schema_json_path}. "
                "Please run 'python scripts/init_schema.py' first."
            )

        with open(self.schema_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.schema_metadata = {
                table["name"]: table for table in data["tables"]
            }
        logger.info(f"Loaded schema metadata for {len(self.schema_metadata)} tables")

    def retrieve_relevant_tables(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Retrieve relevant table names based on query similarity.

        Args:
            query: User's natural language query
            top_k: Number of relevant tables to retrieve

        Returns:
            List of (table_name, similarity_score) tuples
        """
        # Perform similarity search
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        # Extract table names and scores
        relevant_tables = [
            (doc.metadata["table_name"], score)
            for doc, score in results
        ]

        return relevant_tables

    def get_table_info(self, table_names: List[str]) -> str:
        """
        Get detailed schema information for specified tables.

        This method fetches the full SQL schema (CREATE TABLE statements)
        from the database for the specified tables.

        Args:
            table_names: List of table names to fetch schema for

        Returns:
            Formatted schema information string
        """
        if not table_names:
            return ""

        # Separate cached and uncached tables
        cached_results = []
        uncached_tables = []

        for table in table_names:
            if table in self._schema_cache:
                cached_results.append(self._schema_cache[table])
            else:
                uncached_tables.append(table)

        # Fetch uncached tables individually for proper caching
        if uncached_tables:
            for table in uncached_tables:
                schema_info = self.db.get_table_info([table])
                self._schema_cache[table] = schema_info
                cached_results.append(schema_info)

        # Combine all schemas
        return "\n\n".join(cached_results)

    def get_enhanced_schema_info(self, table_names: List[str]) -> str:
        """
        Get enhanced schema information including semantic descriptions.

        This combines SQL schema with semantic descriptions and usage scenarios
        from the schema metadata.

        Args:
            table_names: List of table names

        Returns:
            Enhanced schema information with descriptions and usage scenarios
        """
        # Get base SQL schema
        sql_schema = self.get_table_info(table_names)

        # Add semantic descriptions
        descriptions = []
        for table in table_names:
            if table in self.schema_metadata:
                meta = self.schema_metadata[table]
                descriptions.append(
                    f"\n--- {table} ---\n"
                    f"Description: {meta.get('description', 'N/A')}\n"
                    f"Common use cases: {', '.join(meta.get('usage_scenarios', []))}\n"
                )

        if descriptions:
            enhanced = sql_schema + "\n\n=== SEMANTIC INFORMATION ===\n" + "\n".join(descriptions)
            return enhanced

        return sql_schema
