#!/usr/bin/env python3
"""
Build FAISS vector index from schema descriptions.

This script:
1. Loads schema_descriptions.json
2. Converts table schemas into documents suitable for retrieval
3. Generates embeddings using AWS Bedrock
4. Builds and saves FAISS index

Usage:
    python scripts/build_index.py

Configuration:
    Reads from .env file:
    - SCHEMA_OUTPUT_PATH: Input schema JSON path (default: ./data/schema_descriptions.json)
    - INDEX_OUTPUT_PATH: FAISS index output path (default: ./data/faiss_index)
    - EMBEDDING_MODEL_ID: Bedrock embedding model (default: amazon.titan-embed-text-v2:0)
"""

import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()


def load_schema_descriptions(json_path: str) -> dict:
    """Load schema descriptions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_documents(schema_data: dict) -> List[Document]:
    """Convert schema data into LangChain Document objects for indexing."""
    documents = []

    for table in schema_data["tables"]:
        # Skip tables with "_" in their name
        if "_" in table["name"]:
            print(f"  ⏭️  Skipped table (contains '_'): {table['name']}")
            continue
        # Build comprehensive document content
        content_parts = [
            f"Table: {table['name']}",
            f"Description: {table['description']}",
            f"Columns: {', '.join(table['columns'])}",
        ]

        # Add usage scenarios
        if table.get("usage_scenarios"):
            scenarios = ', '.join(table["usage_scenarios"])
            content_parts.append(f"Use cases: {scenarios}")

        # Add relationships
        if table.get("relationships"):
            content_parts.append(f"Relationships: {table['relationships']}")

        # Add foreign key details
        if table.get("foreign_keys"):
            fk_lines = [
                f"{fk['from_column']} links to {fk['to_table']}.{fk['to_column']}"
                for fk in table["foreign_keys"]
            ]
            content_parts.append(f"Foreign Keys: {'; '.join(fk_lines)}")

        # Create document
        doc = Document(
            page_content="\n".join(content_parts),
            metadata={
                "table_name": table["name"],
                "columns": table["columns"],
                "column_types": table.get("column_types", {}),
                "foreign_keys": table.get("foreign_keys", [])
            }
        )

        documents.append(doc)
        print(f"  📄 Created document for table: {table['name']}")

    return documents


def build_faiss_index(documents: List[Document], embeddings_model) -> FAISS:
    """Build FAISS vector store from documents."""
    print(f"\n🔨 Building FAISS index with {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings_model)
    print(f"  ✅ FAISS index created successfully")
    return vectorstore


def main():
    """Main execution flow."""
    # Load configuration from environment variables
    schema_json_path = os.getenv("SCHEMA_OUTPUT_PATH", "./data/schema_descriptions.json")
    index_output_path = os.getenv("INDEX_OUTPUT_PATH", "./data/faiss_index")
    embedding_model_id = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

    print("🚀 Starting FAISS index building process...")
    print(f"  📄 Schema input: {schema_json_path}")
    print(f"  📁 Index output: {index_output_path}")
    print(f"  🤖 Embedding model: {embedding_model_id}")

    # Check if schema JSON exists
    if not Path(schema_json_path).exists():
        print(f"\n❌ Error: {schema_json_path} not found!")
        print("   Please run 'python scripts/init_schema.py' first.")
        return

    # Load schema data
    print(f"\n📖 Loading schema descriptions from: {schema_json_path}")
    schema_data = load_schema_descriptions(schema_json_path)
    print(f"  ✅ Loaded {len(schema_data['tables'])} tables")

    # Create documents
    print(f"\n📝 Converting schemas to documents...")
    documents = create_documents(schema_data)

    # Initialize embeddings model
    print(f"\n🤖 Initializing Bedrock embeddings (model: {embedding_model_id})...")
    embeddings = BedrockEmbeddings(model_id=embedding_model_id)

    # Build FAISS index
    vectorstore = build_faiss_index(documents, embeddings)

    # Save index to disk
    print(f"\n💾 Saving FAISS index to: {index_output_path}")
    Path(index_output_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_output_path)

    print(f"\n✅ FAISS index built and saved successfully!")
    print(f"📦 Index location: {index_output_path}")
    print(f"📊 Total documents indexed: {len(documents)}")

    # Test retrieval
    print(f"\n🧪 Testing retrieval with sample query...")
    test_query = "Which tables contain customer information?"
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"  Query: '{test_query}'")
    print(f"  Top 3 results:")
    for i, doc in enumerate(results, 1):
        table_name = doc.metadata.get("table_name", "Unknown")
        print(f"    {i}. {table_name}")


if __name__ == "__main__":
    main()
