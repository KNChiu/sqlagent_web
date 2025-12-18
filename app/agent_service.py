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
import json
import re

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from .parsers import extract_tool_calls_data, parse_chart_data
from .debug_utils import print_agent_reasoning, detect_phase_from_message
from .schema_rag import SchemaRAGService
from .custom_tools import create_rag_sql_tools
from .middleware import SafetyMiddleware
from .config import settings
from .prompts import build_main_system_message
from .subagents import create_subagents

logger = logging.getLogger(__name__)


class SQLAgentService:
    """Service for managing SQL Agent and processing queries"""

    def __init__(
        self,
        db_uri: str = None,
        model: str = None,
        dialect: str = None,
        query_limit: int = None,
        rag_top_k: int = None
    ):
        """Initialize SQL Agent with LLM, database, and RAG service

        Args:
            db_uri: Database connection URI
            model: LLM model identifier
            dialect: SQL dialect name (auto-detected from db_uri if not provided)
            query_limit: Default result limit for SQL queries
            rag_top_k: Number of relevant tables to retrieve via RAG (max 5)
        """
        # Use environment variables from settings if not provided
        self.db_uri = db_uri or settings.db_uri
        self.model = model or settings.model_name
        self.dialect = dialect or settings.dialect
        self.query_limit = query_limit if query_limit is not None else settings.query_limit
        self.rag_top_k = rag_top_k if rag_top_k is not None else settings.rag_top_k

        # Initialize system prompt
        self.system_message = build_main_system_message(self.dialect, self.query_limit)

        # Initialize LLM with optional authentication parameters
        llm_kwargs = {}
        if settings.model_base_url:
            llm_kwargs["base_url"] = settings.model_base_url
            logger.info(f"Using custom base_url: {settings.model_base_url}")
        if settings.model_api_key:
            llm_kwargs["api_key"] = settings.model_api_key
            logger.info("Using custom API key from MODEL_API_KEY environment variable")

        self.llm = init_chat_model(model=self.model, **llm_kwargs)
        logger.info(f"LLM initialized: {self.model}")

        # Initialize database
        self.db = SQLDatabase.from_uri(
            self.db_uri,
            view_support=True,
            sample_rows_in_table_info=3,
            lazy_table_reflection=True  # Lazy load table metadata for faster connection
        )

        # Initialize RAG service (reuse db instance to avoid duplicate connections)
        logger.info("Initializing Schema RAG Service...")
        self.rag_service = SchemaRAGService(
            db=self.db,
            db_uri=self.db_uri,
            index_path=settings.index_path,
            schema_json_path=settings.schema_json_path
        )

        # Initialize middleware and tools
        logger.info("Initializing DeepAgent middleware and tools...")
        # Create RAG-enhanced SQL tools directly (no need for middleware wrapper)
        rag_tools = create_rag_sql_tools(
            rag_service=self.rag_service,
            db=self.db,
            rag_top_k=self.rag_top_k,
            query_limit=self.query_limit
        )
        safety_middleware = SafetyMiddleware(query_limit=self.query_limit)

        # Create subagents if enabled
        subagents = []
        if settings.enable_subagents:
            logger.info("Configuring subagents...")
            subagents = create_subagents(
                rag_service=self.rag_service,
                rag_top_k=self.rag_top_k,
                subagent_model=settings.subagent_model
            )
        else:
            logger.info("Subagents disabled - running in simple mode")

        # Create DeepAgent with middleware and subagents
        logger.info(f"Creating DeepAgent with middleware and {len(subagents)} subagent(s)...")
        self.agent = create_deep_agent(
            model=self.llm,  # Pass initialized chat model object
            tools=rag_tools,  # Provide tools directly
            middleware=[
                safety_middleware,
                # TodoListMiddleware is included by default
                # FilesystemMiddleware is included by default
            ],
            subagents=subagents,  # Stage 2: enable subagents
            system_prompt=self.system_message
        )
        logger.info(f"DeepAgent initialized successfully (model: {self.model}, subagents: {len(subagents)})")

    def _enhance_data_with_llm(
        self,
        user_query: str,
        sql_query: str,
        query_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze query results and recommend optimal chart configuration.

        Args:
            user_query: Original user question in natural language
            sql_query: Executed SQL query
            query_results: Query results as List[Dict]

        Returns:
            Chart.js compatible chart_data dict, or None if analysis fails
        """
        try:
            # Step 1: Prepare sample data (first 3 rows to reduce token usage)
            sample_size = min(3, len(query_results))
            sample_data = query_results[:sample_size]
            column_names = list(sample_data[0].keys()) if sample_data else []

            # Step 2: Build LLM analysis prompt (embedded in function)
            prompt = f"""You are a data visualization expert. Analyze the following query results and recommend the optimal chart configuration.

User Question: "{user_query}"
SQL Query: {sql_query}
Columns: {', '.join(column_names)}
Sample Data (first {sample_size} rows): {json.dumps(sample_data, ensure_ascii=False)}

Task: Recommend chart configuration.

Requirements:
1. Chart type: "bar" (categorical), "line" (time series), "pie" (distribution), "table" (no numeric data)
2. Select: label_column (X-axis), value_columns (Y-axis, can be multiple)
3. Sort: sort_by, sort_order ("asc"/"desc")
4. Insights: 1-2 sentences summary
5. Confidence (0-1): Return < 0.5 if unsuitable for visualization

Output JSON:
{{
  "chart_type": "bar",
  "label_column": "Country",
  "value_columns": ["Total"],
  "sort_by": "Total",
  "sort_order": "desc",
  "insights": "USA has highest sales",
  "confidence": 0.9
}}

CRITICAL: Return ONLY valid JSON."""

            # Step 3: Call LLM (synchronous version)
            logger.info("Starting LLM data analysis...")
            response = self.llm.invoke([("user", prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Step 4: Parse JSON response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try direct JSON parsing
                json_str = response_text.strip()

            recommendation = json.loads(json_str)

            # Step 5: Validate confidence
            confidence = recommendation.get('confidence', 0)
            if confidence < 0.5:
                logger.warning(f"LLM confidence too low ({confidence}), using fallback")
                return None

            # Step 6: Build chart_data from recommendation
            chart_type = recommendation['chart_type']
            label_column = recommendation['label_column']
            value_columns = recommendation['value_columns']
            sort_by = recommendation.get('sort_by')
            sort_order = recommendation.get('sort_order', 'desc')

            # Sort data if specified
            sorted_results = query_results
            if sort_by and sort_by in query_results[0]:
                reverse = (sort_order == 'desc')
                sorted_results = sorted(
                    query_results,
                    key=lambda x: x.get(sort_by, 0),
                    reverse=reverse
                )

            # Limit to 10 data points for chart clarity
            sorted_results = sorted_results[:10]

            # Extract labels
            labels = [str(row.get(label_column, '')) for row in sorted_results]

            # Extract values (support multiple datasets)
            datasets = []
            colors = [
                'rgba(59, 130, 246, 0.5)',   # Blue
                'rgba(16, 185, 129, 0.5)',   # Green
                'rgba(249, 115, 22, 0.5)',   # Orange
            ]

            for idx, value_col in enumerate(value_columns):
                values = []
                for row in sorted_results:
                    value = row.get(value_col)
                    # Convert to float
                    try:
                        if isinstance(value, (int, float)):
                            values.append(float(value))
                        elif isinstance(value, str):
                            # Remove currency symbols and convert
                            clean_value = re.sub(r'[^\d.]', '', value)
                            values.append(float(clean_value) if clean_value else 0)
                        else:
                            values.append(0)
                    except (ValueError, TypeError):
                        values.append(0)

                color = colors[idx % len(colors)]
                datasets.append({
                    'label': value_col,
                    'data': values,
                    'backgroundColor': color,
                    'borderColor': color.replace('0.5', '1'),
                    'borderWidth': 2
                })

            # Build final chart_data
            chart_data = {
                'type': chart_type,
                'data': {
                    'labels': labels,
                    'datasets': datasets
                }
            }

            logger.info(f"LLM-enhanced chart created (type: {chart_type}, confidence: {confidence:.2f})")
            return chart_data

        except json.JSONDecodeError as e:
            logger.warning(f"LLM response JSON parsing failed: {e}")
            return None
        except KeyError as e:
            logger.warning(f"Missing required field in LLM response: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {type(e).__name__}: {e}")
            return None

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

        logger.info(f"Agent execution completed in {agent_duration:.2f}s ({len(messages)} messages)")

        # Display agent reasoning process
        print_agent_reasoning(messages, query)

        # Extract final answer
        final_message = messages[-1]
        response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)

        # Extract SQL query and structured results
        sql_query, query_results = extract_tool_calls_data(messages)

        if sql_query:
            logger.info(f"Extracted SQL: {sql_query[:150]}...")

        # Parse chart data
        chart_data = None

        # Try LLM-enhanced data analysis first
        if query_results and isinstance(query_results, list) and len(query_results) > 0:
            try:
                chart_data = self._enhance_data_with_llm(query, sql_query or "", query_results)
                if chart_data:
                    logger.info("Using LLM-enhanced chart")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")

        # Fallback to original parse_chart_data()
        if not chart_data:
            if query_results:
                chart_data = parse_chart_data(query_results, sql_query or "")
                logger.info("Using fallback chart parsing")
            elif response_text:
                chart_data = parse_chart_data(response_text, sql_query or "")

        # Prepare raw data for table view
        raw_data = None
        if query_results and isinstance(query_results, list):
            raw_data = query_results

        # Calculate performance metrics and execution statistics
        request_end_time = time.time()
        total_duration = request_end_time - request_start_time
        data_processing_duration = total_duration - agent_duration

        # Count subagent calls and tool calls for monitoring
        subagent_calls = 0
        tool_calls = 0

        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', '')
                    tool_calls += 1

                    if 'task' in tool_name.lower():
                        subagent_calls += 1
                        subagent_name = tool_call.get('args', {}).get('name', 'unknown')
                        logger.info(f"Subagent delegation: {subagent_name}")

        # Display performance summary with new metrics
        self._print_performance_summary(
            agent_duration,
            data_processing_duration,
            total_duration,
            subagent_calls,
            tool_calls
        )

        return {
            "message": response_text,
            "sql": sql_query,
            "chart_data": chart_data,
            "raw_data": raw_data,
            "performance": {
                "agent_time": agent_duration,
                "data_processing_time": data_processing_duration,
                "total_time": total_duration,
                "subagent_calls": subagent_calls,
                "tool_calls": tool_calls,
                "success": True
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

        # Log streaming start
        logger.info(f"Starting STREAMING query: '{query}'")

        # Send initial status
        yield {"type": "start", "message": "🚀 啟動 Agent..."}
        await asyncio.sleep(0)

        # Track execution phases
        current_phase = None
        all_messages = []
        stream_start_time = time.time()

        # Track tool call descriptions for result display
        tool_descriptions = {}  # {tool_call_id: description}

        # Stream agent execution with recursion limit
        # DeepAgent uses astream() with stream_mode="values"
        async for chunk in self.agent.astream(
            {"messages": input_messages},
            stream_mode="values",
            config={"recursion_limit": settings.recursion_limit}
        ):
            # DeepAgent returns chunks with "messages" key directly
            if "messages" in chunk:
                messages = chunk["messages"]

                # Accumulate all messages
                all_messages = messages  # Update full message list

                if messages:
                    latest_msg = messages[-1]

                    # Track tool call descriptions (AIMessage with tool_calls)
                    if hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                        for tool_call in latest_msg.tool_calls:
                            tool_call_id = tool_call.get('id')
                            tool_args = tool_call.get('args', {})
                            description = tool_args.get('description', '')
                            if tool_call_id and description:
                                tool_descriptions[tool_call_id] = description

                    # Detect phase from message
                    phase_info = detect_phase_from_message(latest_msg)

                    # Enhance result message with description (ToolMessage)
                    if hasattr(latest_msg, 'tool_call_id') and latest_msg.tool_call_id in tool_descriptions:
                        description = tool_descriptions[latest_msg.tool_call_id]
                        if phase_info["phase"] == "processing_results":
                            # Prepend description to result message
                            phase_info["message"] = f"{description} → {phase_info['message']}"

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

        # Calculate streaming duration
        stream_duration = time.time() - stream_start_time

        # Extract final results
        sql_query, query_results = extract_tool_calls_data(all_messages)

        # Count tool calls and subagent calls for streaming
        tool_calls = 0
        subagent_calls = 0
        for msg in all_messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', '')
                    tool_calls += 1
                    if 'task' in tool_name.lower():
                        subagent_calls += 1

        # Parse chart data
        chart_data = None

        # Try LLM-enhanced data analysis first
        if query_results and isinstance(query_results, list) and len(query_results) > 0:
            try:
                chart_data = self._enhance_data_with_llm(query, sql_query or "", query_results)
                if chart_data:
                    logger.info("Using LLM-enhanced chart")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")

        # Fallback to original parse_chart_data()
        if not chart_data and query_results:
            chart_data = parse_chart_data(query_results, sql_query or "")
            logger.info("Using fallback chart parsing")

        # Prepare raw data
        raw_data = None
        if query_results and isinstance(query_results, list):
            raw_data = query_results

        # Get final response text
        final_message = all_messages[-1] if all_messages else None
        response_text = final_message.content if (final_message and hasattr(final_message, 'content')) else ""

        # Log summary
        rows = len(query_results) if isinstance(query_results, list) else 0
        logger.info(f"Streaming completed in {stream_duration:.2f}s - SQL: {bool(sql_query)}, Rows: {rows}, Tools: {tool_calls}, Subagents: {subagent_calls}")

        # Send completion event
        final_event = {
            "type": "complete",
            "message": response_text,
            "sql": sql_query,
            "chart_data": chart_data,
            "raw_data": raw_data
        }
        yield final_event

    def _print_performance_summary(
        self,
        agent_time: float,
        data_time: float,
        total_time: float,
        subagent_calls: int = 0,
        tool_calls: int = 0
    ):
        """Log performance summary for debugging"""
        logger.info(f"Performance - Total: {total_time:.2f}s, Agent: {agent_time:.2f}s, Data: {data_time:.2f}s, Tools: {tool_calls}, Subagents: {subagent_calls}")
