#!/usr/bin/env python3
"""
Initialize schema descriptions using LLM.

This script:
1. Connects to database and extracts all table schemas
2. Uses Claude via AWS Bedrock to generate semantic descriptions for each table
3. Outputs schema_descriptions.json for RAG indexing

Usage:
    python scripts/init_schema.py

Configuration:
    Reads from .env file:
    - DB_URI: Database connection URI (default: sqlite:///./Chinook.db)
    - MODEL_NAME: LLM model identifier (default: bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0)
    - SCHEMA_OUTPUT_PATH: Output JSON path (default: ./data/schema_descriptions.json)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from sqlalchemy import create_engine, inspect

# Load environment variables from .env file
load_dotenv()


def get_table_names(db_uri: str) -> List[str]:
    """Extract all table names from database using SQLAlchemy Inspector."""
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())
    engine.dispose()
    return tables


def get_table_schema(db_uri: str, table_name: str) -> Dict[str, Any]:
    """Extract detailed schema information for a specific table using SQLAlchemy Inspector."""
    engine = create_engine(db_uri)
    inspector = inspect(engine)

    # Get column information
    columns = inspector.get_columns(table_name)

    # Get primary key information
    pk_constraint = inspector.get_pk_constraint(table_name)
    pk_columns = set(pk_constraint.get('constrained_columns', []))

    # Get foreign key information
    foreign_keys = inspector.get_foreign_keys(table_name)

    engine.dispose()

    # Format column info
    column_info = [
        {
            "name": col["name"],
            "type": str(col["type"]),
            "not_null": not col.get("nullable", True),
            "primary_key": col["name"] in pk_columns
        }
        for col in columns
    ]

    # Format foreign key info
    fk_info = [
        {
            "from_column": fk["constrained_columns"][0] if fk["constrained_columns"] else "",
            "to_table": fk["referred_table"],
            "to_column": fk["referred_columns"][0] if fk["referred_columns"] else ""
        }
        for fk in foreign_keys
    ]

    return {
        "name": table_name,
        "columns": column_info,
        "foreign_keys": fk_info
    }


def generate_table_description(llm: ChatBedrockConverse, schema: Dict[str, Any]) -> str:
    """Use Claude to generate semantic description for a table."""
    table_name = schema["name"]
    columns = [col["name"] for col in schema["columns"]]
    foreign_keys = schema["foreign_keys"]

    # Build FK relationships description
    fk_desc = ""
    if foreign_keys:
        fk_lines = [f"  - {fk['from_column']} → {fk['to_table']}.{fk['to_column']}" for fk in foreign_keys]
        fk_desc = "\nForeign Keys:\n" + "\n".join(fk_lines)

    prompt = f"""Analyze this database table and provide a concise semantic description.

Table: {table_name}
Columns: {', '.join(columns)}{fk_desc}

Provide:
1. A 1-2 sentence description of what this table stores
2. Common query scenarios where this table would be relevant (3-5 keywords)
3. Relationships to other tables (if foreign keys exist)

Format your response as:
DESCRIPTION: [your description]
SCENARIOS: [comma-separated keywords]
RELATIONSHIPS: [brief explanation of how this table relates to others]

Keep it concise and focused on information retrieval use cases."""

    response = llm.invoke(prompt)
    return response.content


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response into structured format."""
    lines = response.strip().split('\n')
    result = {
        "description": "",
        "usage_scenarios": [],
        "relationships": ""
    }

    current_key = None
    for line in lines:
        line = line.strip()
        if line.startswith("DESCRIPTION:"):
            current_key = "description"
            result["description"] = line.replace("DESCRIPTION:", "").strip()
        elif line.startswith("SCENARIOS:"):
            current_key = "usage_scenarios"
            scenarios_text = line.replace("SCENARIOS:", "").strip()
            result["usage_scenarios"] = [s.strip() for s in scenarios_text.split(",")]
        elif line.startswith("RELATIONSHIPS:"):
            current_key = "relationships"
            result["relationships"] = line.replace("RELATIONSHIPS:", "").strip()
        elif current_key and line:
            # Continue previous section
            result[current_key] += " " + line

    return result


def extract_model_id_from_name(model_name: str) -> str:
    """Extract AWS Bedrock model ID from model name.

    Converts 'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    to 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    """
    if model_name.startswith("bedrock:"):
        return model_name.replace("bedrock:", "")
    return model_name


def load_existing_schema(output_path: str) -> Dict[str, Any]:
    """Load existing schema file if exists, otherwise return empty structure."""
    output_file = Path(output_path)

    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not read existing schema file: {e}")
            print(f"  Starting fresh...")
            return {"tables": []}

    return {"tables": []}


def save_schema_incremental(output_path: str, schema_data: Dict[str, Any]) -> None:
    """Save schema data incrementally to file after each table processing."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"❌ Error: Could not save schema file: {e}")
        raise


def main():
    """Main execution flow with incremental processing and error recovery."""
    # Load configuration from environment variables
    db_uri = os.getenv("DB_URI", "sqlite:///./Chinook.db")
    output_path = os.getenv("SCHEMA_OUTPUT_PATH", "./data/schema_descriptions.json")
    model_name = os.getenv("MODEL_NAME", "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    # Extract model ID
    model_id = extract_model_id_from_name(model_name)

    print("🔍 Initializing schema extraction...")
    print(f"  📁 Database URI: {db_uri}")
    print(f"  🤖 Model: {model_id}")
    print(f"  📄 Output: {output_path}")

    # Load existing schema (if any)
    schema_data = load_existing_schema(output_path)
    processed_tables = {table["name"] for table in schema_data["tables"]}

    if processed_tables:
        print(f"📋 Found existing schema with {len(processed_tables)} tables already processed")
        print(f"  Already processed: {', '.join(sorted(processed_tables))}")

    # Initialize LLM
    llm = ChatBedrockConverse(model=model_id, temperature=0)

    # Get all tables
    tables = get_table_names(db_uri)
    print(f"📊 Found {len(tables)} tables in database: {', '.join(tables)}")

    # Filter out already processed tables
    remaining_tables = [t for t in tables if t not in processed_tables]

    if not remaining_tables:
        print(f"\n✅ All tables already processed! Nothing to do.")
        return

    print(f"🎯 Need to process {len(remaining_tables)} remaining tables: {', '.join(remaining_tables)}")

    # Process each remaining table with error handling
    success_count = 0
    error_count = 0

    for table in remaining_tables:
        print(f"\n🔨 Processing table: {table}")

        try:
            # Extract schema
            schema = get_table_schema(db_uri, table)

            # Generate description using LLM
            print(f"  🤖 Generating semantic description...")
            llm_response = generate_table_description(llm, schema)
            parsed = parse_llm_response(llm_response)

            # Combine schema and description
            table_data = {
                "name": table,
                "description": parsed["description"],
                "columns": [col["name"] for col in schema["columns"]],
                "column_types": {col["name"]: col["type"] for col in schema["columns"]},
                "usage_scenarios": parsed["usage_scenarios"],
                "relationships": parsed["relationships"],
                "foreign_keys": schema["foreign_keys"]
            }

            # Add to schema data
            schema_data["tables"].append(table_data)

            # Save incrementally after each table
            save_schema_incremental(output_path, schema_data)

            print(f"  ✅ Description: {parsed['description'][:80]}...")
            print(f"  💾 Saved to file (progress: {len(schema_data['tables'])}/{len(tables)})")
            success_count += 1

        except Exception as e:
            print(f"  ❌ Error processing table {table}: {e}")
            print(f"  ⏩ Skipping to next table...")
            error_count += 1
            continue

    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ Schema extraction completed!")
    print(f"  📄 Output file: {output_path}")
    print(f"  📊 Total tables in database: {len(tables)}")
    print(f"  ✅ Successfully processed: {success_count}")
    print(f"  ⚠️  Already processed (skipped): {len(processed_tables)}")
    if error_count > 0:
        print(f"  ❌ Failed: {error_count}")
    print(f"  📦 Total in output file: {len(schema_data['tables'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
