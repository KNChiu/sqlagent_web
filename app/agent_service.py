"""SQL Agent Service

This module provides the SQLAgentService class that encapsulates:
- Agent initialization with LLM and database
- Query processing with conversation history
- Streaming query execution with real-time status updates
"""

from typing import List, Dict, Any, Optional
import time
import asyncio
import logging

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from .parsers import extract_tool_calls_data, parse_chart_data
from .debug_utils import print_agent_reasoning, detect_phase_from_message
from .schema_rag import SchemaRAGService
from .custom_tools import create_rag_sql_tools
from .config import settings

logger = logging.getLogger(__name__)


# System prompt template for the SQL Agent (RAG-optimized)
PROMPT_TEMPLATE = """system
You are an agent designed to interact with a SQL database using RAG-enhanced schema retrieval.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
However, if the user explicitly requests a different number (e.g., "show me 10 rows", "give me 20 records"), you MUST respect their request and override the default limit.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

IMPORTANT: Use the sql_db_schema_rag tool FIRST to get relevant table schemas based on your query understanding.
The RAG tool will automatically find the most relevant tables - you do NOT need to list all tables first.
Simply describe what data you need in natural language, and the tool will return the appropriate schemas.

IMPORTANT: When the user asks follow-up questions (e.g., "show me more", "what about Canada?", "give me 10 records instead"),
you MUST refer to the conversation history to understand the context. Pay attention to previous queries and results in this conversation.

EFFICIENCY RULES (CRITICAL):
1. Once you execute a query and get valid results, STOP immediately and provide your answer.
2. Do NOT rewrite queries just to change output format, column names, or data presentation.
3. Do NOT execute the same query multiple times with minor variations.
4. If you get data back from sql_db_query, that means the query was successful - answer the user's question directly.
5. Only retry if you get an ERROR message. Valid results (even if formatted differently than expected) should be used immediately."""


class SQLAgentService:
    """Service for managing SQL Agent and processing queries"""

    def __init__(
        self,
        db_uri: str = None,
        model: str = None,
        dialect: str = None,
        top_k: int = None,
        rag_top_k: int = None
    ):
        """Initialize SQL Agent with LLM, database, and RAG service

        Args:
            db_uri: Database connection URI
            model: LLM model identifier
            dialect: SQL dialect name (auto-detected from db_uri if not provided)
            top_k: Default result limit for SQL queries
            rag_top_k: Number of relevant tables to retrieve via RAG (max 5)
        """
        # Use environment variables from settings if not provided
        self.db_uri = db_uri or settings.db_uri
        self.model = model or settings.model_name
        self.dialect = dialect or settings.dialect
        self.top_k = top_k if top_k is not None else settings.top_k
        self.rag_top_k = rag_top_k if rag_top_k is not None else settings.rag_top_k

        # Initialize system prompt
        self.system_message = PROMPT_TEMPLATE.format(dialect=self.dialect, top_k=self.top_k)

        # Initialize LLM and database
        # Build kwargs for init_chat_model with optional authentication parameters
        llm_kwargs = {}

        if settings.model_base_url:
            llm_kwargs["base_url"] = settings.model_base_url
            logger.info(f"Using custom base_url: {settings.model_base_url}")

        if settings.model_api_key:
            llm_kwargs["api_key"] = settings.model_api_key
            logger.info("Using custom API key from MODEL_API_KEY environment variable")

        self.llm = init_chat_model(model=self.model, **llm_kwargs)
        self.db = SQLDatabase.from_uri(
            self.db_uri,
            view_support=True,
            sample_rows_in_table_info=3,
            lazy_table_reflection=True  # Lazy load table metadata for faster connection
        )

        # Initialize RAG service (reuse db instance to avoid duplicate connections)
        logger.info("Initializing Schema RAG Service...")
        self.rag_service = SchemaRAGService(db=self.db, db_uri=self.db_uri)

        # Create RAG-enabled tools
        self.tools = create_rag_sql_tools(
            rag_service=self.rag_service,
            db=self.db,
            top_k=self.rag_top_k
        )

        # Create agent with RAG tools
        self.agent = create_agent(self.llm, self.tools, system_prompt=self.system_message)
        logger.info(f"SQL Agent initialized successfully (model: {self.model}, tools: {len(self.tools)})")

    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process a query and return structured results

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages

        Returns:
            Dict with keys: message, sql, chart_data, raw_data, performance
        """
        # Record start time
        request_start_time = time.time()

        # Build message history
        input_messages = []

        # Convert conversation history to LangChain format
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role and content:
                    langchain_role = "user" if role == "user" else "ai"
                    input_messages.append((langchain_role, content))

        # Add current query
        input_messages.append(("user", query))

        # Invoke agent with recursion limit
        logger.info(f"Starting agent execution for query: '{query}'")
        agent_start_time = time.time()
        result = self.agent.invoke(
            {"messages": input_messages},
            config={"recursion_limit": settings.recursion_limit}
        )
        agent_end_time = time.time()
        agent_duration = agent_end_time - agent_start_time

        messages = result.get("messages", [])
        if not messages:
            logger.error("No response from agent")
            raise ValueError("No response from agent")

        logger.info(f"Agent execution completed in {agent_duration:.2f}s")

        # Display agent reasoning process
        print_agent_reasoning(messages, query)

        # Extract final answer
        final_message = messages[-1]
        response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)

        # Extract SQL query and structured results
        sql_query, query_results = extract_tool_calls_data(messages)

        # Display data extraction results
        self._print_extraction_results(sql_query, query_results)

        # Parse chart data
        chart_data = None
        if query_results:
            chart_data = parse_chart_data(query_results, sql_query or "")

        # Fallback: try to parse from response text
        if not chart_data and response_text:
            chart_data = parse_chart_data(response_text, sql_query or "")

        # Log data extraction results
        logger.info(
            f"Data extraction completed "
            f"(SQL: {'Yes' if sql_query else 'No'}, "
            f"Results: {len(query_results) if isinstance(query_results, list) else 'N/A'} rows, "
            f"Chart: {'Yes' if chart_data else 'No'})"
        )

        # Prepare raw data for table view
        raw_data = None
        if query_results and isinstance(query_results, list):
            raw_data = query_results

        # Calculate performance metrics
        request_end_time = time.time()
        total_duration = request_end_time - request_start_time
        data_processing_duration = total_duration - agent_duration

        # Display performance summary
        self._print_performance_summary(agent_duration, data_processing_duration, total_duration)

        return {
            "message": response_text,
            "sql": sql_query,
            "chart_data": chart_data,
            "raw_data": raw_data,
            "performance": {
                "agent_time": agent_duration,
                "data_processing_time": data_processing_duration,
                "total_time": total_duration
            }
        }

    async def stream_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """Stream query execution with real-time status updates

        Args:
            query: User's natural language query
            conversation_history: Previous conversation messages

        Yields:
            Dicts with event data (type, phase, message, etc.)
        """
        # Build message history
        input_messages = []

        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role and content:
                    langchain_role = "user" if role == "user" else "ai"
                    input_messages.append((langchain_role, content))

        # Add current query
        input_messages.append(("user", query))

        # Send initial status
        yield {"type": "start", "message": "🚀 啟動 Agent..."}
        await asyncio.sleep(0)

        # Track execution phases
        current_phase = None
        all_messages = []

        # Stream agent execution with recursion limit
        for chunk in self.agent.stream(
            {"messages": input_messages},
            stream_mode="updates",
            config={"recursion_limit": settings.recursion_limit}
        ):
            for node_name, data in chunk.items():
                messages = data.get("messages", [])
                if not messages:
                    continue

                # Accumulate all messages
                all_messages.extend(messages)

                latest_msg = messages[-1]

                # Detect phase from message
                phase_info = detect_phase_from_message(latest_msg)

                # Send phase transition event if phase changed
                if phase_info["phase"] != current_phase:
                    current_phase = phase_info["phase"]
                    event_data = {
                        "type": "phase",
                        "phase": phase_info["phase"],
                        "icon": phase_info["icon"],
                        "message": phase_info["message"]
                    }
                    yield event_data
                    await asyncio.sleep(0)

                # Send step detail event
                step_event = {
                    "type": "step",
                    "phase": phase_info["phase"],
                    "details": phase_info["details"]
                }
                yield step_event
                await asyncio.sleep(0)

        # Extract final results
        sql_query, query_results = extract_tool_calls_data(all_messages)

        # Parse chart data
        chart_data = None
        if query_results:
            chart_data = parse_chart_data(query_results, sql_query or "")

        # Prepare raw data
        raw_data = None
        if query_results and isinstance(query_results, list):
            raw_data = query_results

        # Get final response text
        final_message = all_messages[-1] if all_messages else None
        response_text = final_message.content if (final_message and hasattr(final_message, 'content')) else ""

        # Send completion event
        final_event = {
            "type": "complete",
            "message": response_text,
            "sql": sql_query,
            "chart_data": chart_data,
            "raw_data": raw_data
        }
        yield final_event

    def _print_extraction_results(self, sql_query: Optional[str], query_results: Any):
        """Log data extraction results for debugging"""
        logger.debug("=" * 60)
        logger.debug("DATA EXTRACTION RESULTS")
        logger.debug("=" * 60)

        if sql_query:
            logger.debug(f"SQL Query Extracted: {sql_query}")
        else:
            logger.debug("No SQL query found in agent messages")

        if query_results:
            logger.debug(f"Query Results Extracted - Data Type: {type(query_results).__name__}")

            if isinstance(query_results, list):
                logger.debug(f"Structure: List with {len(query_results)} item(s)")

                if len(query_results) > 0:
                    first_item = query_results[0]
                    first_item_type = type(first_item).__name__
                    logger.debug(f"Item Type: {first_item_type}")

                    if isinstance(first_item, dict):
                        columns = list(first_item.keys())
                        logger.debug(f"Columns ({len(columns)}): {', '.join(columns)}")
                        logger.debug(f"Sample Row: {first_item}")
                    elif isinstance(first_item, tuple):
                        logger.debug(f"Tuple Length: {len(first_item)}")
                        logger.debug(f"Sample Row: {first_item}")
                    else:
                        logger.debug(f"Sample Item: {str(first_item)[:100]}")

                    if len(query_results) > 1:
                        logger.debug(f"... and {len(query_results) - 1} more row(s)")
            else:
                preview = str(query_results)[:200]
                logger.debug(f"Preview: {preview}{'...' if len(str(query_results)) > 200 else ''}")
        else:
            logger.debug("No structured results found")

        logger.debug("=" * 60)

    def _print_performance_summary(self, agent_time: float, data_time: float, total_time: float):
        """Log performance summary for debugging"""
        logger.debug("=" * 60)
        logger.debug("PERFORMANCE SUMMARY")
        logger.debug(f"Agent Processing: {agent_time:.2f}s")
        logger.debug(f"Data Processing:  {data_time:.2f}s")
        logger.debug(f"Total Time:       {total_time:.2f}s")
        logger.debug("=" * 60)
