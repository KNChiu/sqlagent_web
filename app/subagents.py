"""Subagent Configuration for DeepAgent

This module defines subagent configurations for task delegation in the SQL Agent.
Subagents provide specialized capabilities for schema exploration and error recovery.
"""

from typing import List, Dict, Any
import logging

from .custom_tools import RAGSchemaInfoTool
from .schema_rag import SchemaRAGService
from .prompts import SCHEMA_EXPLORER_PROMPT, ERROR_RECOVERY_PROMPT
from .config import settings

logger = logging.getLogger(__name__)


def create_subagents(
    rag_service: SchemaRAGService,
    rag_top_k: int,
    subagent_model: str
) -> List[Dict[str, Any]]:
    """
    Create subagent configurations for task delegation.

    Args:
        rag_service: Initialized SchemaRAGService instance
        rag_top_k: Number of relevant tables to retrieve via RAG
        subagent_model: Model identifier for subagents

    Returns:
        List of subagent configuration dictionaries
    """
    subagents = []

    # Schema Explorer Subagent
    schema_explorer = {
        "name": "schema-explorer",
        "description": """DELEGATE to this agent when you need to:
- Understand database structure ("what tables exist", "show schema", "database layout")
- Find relevant tables for a query
- Explore data relationships and table connections

Keywords that trigger this agent: schema, tables, structure, explore, available, database layout, what data

This agent specializes in database schema analysis using RAG retrieval.""",
        "system_prompt": SCHEMA_EXPLORER_PROMPT,
        "tools": [
            RAGSchemaInfoTool(
                rag_service=rag_service,
                top_k=rag_top_k,
                verbose_output=True
            )
        ],
        "model": subagent_model
    }
    subagents.append(schema_explorer)

    # Error Recovery Subagent
    error_recovery = {
        "name": "error-recovery",
        "description": """DELEGATE to this agent when:
- SQL query execution fails
- Syntax errors occur
- Table or column not found errors
- Type mismatch or constraint errors

Keywords that trigger this agent: error, failed, exception, syntax error, not found, invalid

This agent analyzes errors and suggests fixes.""",
        "system_prompt": ERROR_RECOVERY_PROMPT,
        "tools": [
            RAGSchemaInfoTool(
                rag_service=rag_service,
                top_k=rag_top_k,
                verbose_output=False  # Less verbose for error analysis
            )
        ],
        "model": subagent_model
    }
    subagents.append(error_recovery)

    logger.info(f"Created {len(subagents)} subagents: {[s['name'] for s in subagents]}")
    return subagents
