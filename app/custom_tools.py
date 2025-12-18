"""
Custom LangChain Tools for RAG-based SQL Agent.

This module provides custom tools that integrate with Schema RAG Service
to reduce token usage by retrieving only relevant table schemas.
"""

from typing import Optional, Type
from pydantic import BaseModel, Field
import logging

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.utilities import SQLDatabase

from app.schema_rag import SchemaRAGService

logger = logging.getLogger(__name__)


class RAGSchemaInfoInput(BaseModel):
    """Input schema for RAGSchemaInfoTool."""
    query: str = Field(
        description="Natural language description of what data you want to find. "
        "Be specific about the types of information needed (e.g., 'customer purchases', "
        "'artist albums', 'employee information')."
    )


class RAGSchemaInfoTool(BaseTool):
    """
    Tool that uses RAG to retrieve relevant table schemas.

    This tool replaces the traditional ListSQLDatabaseTool and InfoSQLDatabaseTool
    by using semantic search to find only the relevant tables, significantly
    reducing token usage.
    """

    name: str = "sql_db_schema_rag"
    description: str = (
        "Use this tool to get schema information for database tables relevant to your query. "
        "Input should be a natural language description of what data you need. "
        "The tool will automatically find the relevant tables and return their schemas. "
        "Always use this tool FIRST before writing any SQL queries to understand the available data."
    )
    args_schema: Type[BaseModel] = RAGSchemaInfoInput

    rag_service: SchemaRAGService = Field(exclude=True)
    top_k: int = Field(default=5, exclude=True)
    verbose_output: bool = Field(default=True, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute RAG-based schema retrieval."""
        try:
            logger.info(f"RAG schema retrieval for query: '{query}'")

            # Retrieve relevant tables using RAG
            relevant_tables = self.rag_service.retrieve_relevant_tables(
                query=query,
                top_k=self.top_k
            )

            if not relevant_tables:
                logger.info("No relevant tables found")
                return "No relevant tables found for your query. Please rephrase or try a different description."

            # Extract table names
            table_names = [table for table, score in relevant_tables]
            logger.info(f"Retrieved {len(table_names)} relevant tables: {', '.join(table_names)}")

            # Get enhanced schema information
            schema_info = self.rag_service.get_enhanced_schema_info(table_names)

            # Add retrieval context if verbose
            if self.verbose_output:
                header = f"Found {len(table_names)} relevant tables: {', '.join(table_names)}\n\n"
                return header + schema_info

            return schema_info

        except Exception as e:
            logger.error(f"Error retrieving schema: {str(e)}")
            return f"Error retrieving schema: {str(e)}"


class QuerySQLDatabaseInput(BaseModel):
    """Input schema for QuerySQLDatabaseTool."""
    query: str = Field(
        description="A detailed and correct SQL query to execute against the database. "
        "The query must be syntactically correct and reference only existing tables and columns."
    )
    description: str = Field(
        description="A brief one-line description of what this query does. "
        "This will be displayed to users during query execution."
    )


class QuerySQLDatabaseTool(BaseTool):
    """
    Tool for executing SQL queries against the database.

    This is a custom implementation of the standard QuerySQLDatabaseTool,
    adapted to work with our RAG system.
    """

    name: str = "sql_db_query"
    description: str = (
        "Execute a SQL query against the database and get back the result. "
        "If the query is not correct, an error message will be returned. "
        "If an error is returned, rewrite the query, check the query, and try again. "
        "Only use this tool after getting schema information with sql_db_schema_rag."
    )
    args_schema: Type[BaseModel] = QuerySQLDatabaseInput

    db: SQLDatabase = Field(exclude=True)
    query_limit: int = Field(default=5, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        description: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute SQL query and return results as JSON for better data processing."""
        try:
            if description:
                logger.info(f"Executing SQL query: {description}")
            else:
                logger.info(f"Executing SQL query: {query[:100]}...")

            # Execute query and fetch all results
            result = self.db.run(query, fetch="all")

            # Handle different result types
            if isinstance(result, str):
                logger.info(f"Query executed successfully (result type: string)")
                return result

            # Format results as JSON for structured data processing
            if isinstance(result, list):
                if len(result) == 0:
                    logger.info("Query executed successfully (no results)")
                    return "Query executed successfully. No rows returned."

                # Limit to query_limit
                limited_result = result[:self.query_limit]

                # Convert to JSON format for better parsing
                import json
                json_output = json.dumps(limited_result, ensure_ascii=False, default=str)

                result_summary = f"Query returned {len(result)} row(s)"
                if len(result) > self.query_limit:
                    result_summary += f" (showing first {self.query_limit})"

                logger.info(f"Query executed successfully ({len(result)} rows, returning JSON)")
                # Return JSON array directly for structured parsing
                return json_output

            logger.info(f"Query executed successfully (result type: {type(result).__name__})")
            return str(result)

        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            return f"Error executing query: {str(e)}\n\nPlease check the SQL syntax and table/column names, then try again."

    def _format_results(self, results: list) -> str:
        """Format query results as a readable table."""
        if not results:
            return ""

        # Handle both dict and tuple results
        if isinstance(results[0], dict):
            # Dictionary format
            headers = list(results[0].keys())
            rows = [[str(row.get(h, "NULL")) for h in headers] for row in results]
        elif isinstance(results[0], tuple):
            # Tuple format - generate column numbers
            headers = [f"col_{i}" for i in range(len(results[0]))]
            rows = [[str(val) for val in row] for row in results]
        else:
            # Fallback for other types
            return "\n".join(str(row) for row in results)

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        # Build table
        lines = []

        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Rows
        for row in rows:
            row_line = " | ".join(val.ljust(w) for val, w in zip(row, col_widths))
            lines.append(row_line)

        return "\n".join(lines)


def create_rag_sql_tools(
    rag_service: SchemaRAGService,
    db: SQLDatabase,
    rag_top_k: int = 5,
    query_limit: int = 5
) -> list:
    """
    Create a list of RAG-enabled SQL tools for the agent.

    Args:
        rag_service: Initialized SchemaRAGService instance
        db: SQLDatabase instance
        rag_top_k: Number of relevant tables to retrieve via RAG (default: 5)
        query_limit: Maximum number of rows to return from SQL queries (default: 5)

    Returns:
        List of LangChain tools
    """
    tools = [
        RAGSchemaInfoTool(
            rag_service=rag_service,
            top_k=rag_top_k,
            verbose_output=True
        ),
        QuerySQLDatabaseTool(
            db=db,
            query_limit=query_limit
        )
    ]

    return tools
