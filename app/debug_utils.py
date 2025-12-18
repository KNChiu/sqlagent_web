"""Debugging utilities for SQL Agent

This module provides tools for:
- Detecting execution phases from agent messages
- Printing agent reasoning timeline with detailed step tracking
"""

from typing import List, Any
import re
import time
import logging

logger = logging.getLogger(__name__)


def detect_phase_from_message(message) -> dict:
    """Detect execution phase from agent message for real-time status updates

    Returns dict with: phase, icon, message, details
    """

    # Check for tool calls (Agent is requesting to use a tool)
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = (tool_call.get('name', '') or '').lower()
        tool_args = tool_call.get('args', {})

        if 'list_tables' in tool_name:
            return {
                "phase": "schema_discovery",
                "icon": "🔍",
                "message": "正在查詢資料庫結構...",
                "details": {"tool": tool_name}
            }

        elif 'schema' in tool_name:
            # RAG schema tool uses 'query' parameter (natural language)
            query = tool_args.get('query', '')

            # Traditional schema tool uses 'table_names' parameter
            tables = tool_args.get('table_names', [])

            if query:
                # RAG mode: show the natural language query
                display_text = query[:60] + '...' if len(query) > 60 else query
            elif tables:
                # Traditional mode: show table names
                if isinstance(tables, list):
                    display_text = ', '.join(tables)
                else:
                    display_text = str(tables)
            else:
                display_text = '資料表結構'

            return {
                "phase": "schema_analysis",
                "icon": "📋",
                "message": f"正在搜尋相關資料表: {display_text}",
                "details": {"tool": tool_name, "query": query, "tables": tables}
            }

        elif 'sql_db_query' in tool_name:
            sql_query = tool_args.get('query', '')
            description = tool_args.get('description', '')

            # Use description if provided, otherwise show SQL preview
            if description:
                display_message = description
            else:
                display_message = sql_query[:80] + '...' if len(sql_query) > 80 else sql_query

            return {
                "phase": "query_execution",
                "icon": "⚡",
                "message": display_message,
                "details": {
                    "tool": tool_name,
                    "sql": sql_query,
                    "description": description
                }
            }

        elif 'query_checker' in tool_name or 'check' in tool_name:
            return {
                "phase": "validation",
                "icon": "✓",
                "message": "正在驗證 SQL 語法...",
                "details": {"tool": tool_name}
            }

        else:
            return {
                "phase": "processing",
                "icon": "🔧",
                "message": f"正在執行工具: {tool_name}",
                "details": {"tool": tool_name}
            }

    # Check for tool results (ToolMessage - tool execution completed)
    elif hasattr(message, 'name'):
        tool_name = (getattr(message, 'name', '') or '').lower()
        content = message.content

        if 'sql_db_query' in tool_name:
            # Query execution completed - Enhanced row_count calculation
            if isinstance(content, list):
                row_count = len(content)
            elif isinstance(content, str):
                # Try to parse string as Python literal
                try:
                    import ast
                    parsed_content = ast.literal_eval(content.strip())
                    row_count = len(parsed_content) if isinstance(parsed_content, (list, tuple)) else 0
                except:
                    # Fallback: count tuples using regex
                    row_count = len(re.findall(r'\([^)]+\)', content))
            else:
                row_count = 0

            # Check if there was an error
            if isinstance(content, str) and ('error' in content.lower() or 'exception' in content.lower()):
                return {
                    "phase": "error",
                    "icon": "❌",
                    "message": "SQL 查詢失敗，正在重試...",
                    "details": {"error_preview": content[:100]}
                }

            return {
                "phase": "processing_results",
                "icon": "✓",
                "message": f"查詢結果: {row_count} 筆資料",
                "details": {"row_count": row_count, "content_type": type(content).__name__}
            }

        elif 'list_tables' in tool_name:
            tables = str(content).split(',') if ',' in str(content) else [str(content)]
            table_count = len(tables)
            return {
                "phase": "schema_discovery",
                "icon": "🔍",
                "message": f"找到 {table_count} 個資料表",
                "details": {"table_count": table_count, "tables": tables[:5]}
            }

        elif 'schema' in tool_name:
            # Extract table names from content if available
            content_str = str(content)
            table_names = []

            # Try to extract from "Found N relevant tables: table1, table2, ..."
            match = re.search(r'Found (\d+) relevant tables?:\s*(.+?)(?:\n|$)', content_str)
            if match:
                table_count = match.group(1)
                table_list = match.group(2).strip()
                table_names = [t.strip() for t in table_list.split(',')]

                return {
                    "phase": "schema_analysis",
                    "icon": "📋",
                    "message": f"找到 {table_count} 個相關資料表: {', '.join(table_names[:5])}{'...' if len(table_names) > 5 else ''}",
                    "details": {"table_count": int(table_count), "tables": table_names}
                }

            return {
                "phase": "schema_analysis",
                "icon": "📋",
                "message": "Schema 載入完成",
                "details": {"content_length": len(content_str)}
            }

    # Final answer generation (AIMessage with content, no tool calls)
    elif hasattr(message, 'content') and message.content:
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return {
                "phase": "generating_answer",
                "icon": "💬",
                "message": "正在生成回應...",
                "details": {"content_length": len(message.content)}
            }

    # Default/unknown phase
    return {
        "phase": "processing",
        "icon": "⚙️",
        "message": "處理中...",
        "details": {"message_type": type(message).__name__}
    }


def print_agent_reasoning(messages: List[Any], user_query: str):
    """Parse and display agent's reasoning process in a timeline format"""
    logger.info(f"Agent reasoning timeline for query: '{user_query}'")
    logger.debug("=" * 60)
    logger.debug("AGENT REASONING TIMELINE")
    logger.debug("=" * 60)
    logger.debug(f"User Query: \"{user_query}\"")

    step_counter = 0
    tool_call_times = {}
    sql_attempts = []

    # Phase tracking
    schema_phase = False
    query_phase = False

    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__

        # Track HumanMessage (initial query)
        if msg_type == "HumanMessage" and i == 0:
            continue  # Skip, already displayed above

        # Track AIMessage with tool calls
        if msg_type == "AIMessage" and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                step_counter += 1
                tool_name = tool_call.get('name', 'unknown')
                args = tool_call.get('args', {})

                # Determine phase
                if 'list_tables' in tool_name.lower():
                    if not schema_phase:
                        logger.debug("SCHEMA INSPECTION PHASE")
                        schema_phase = True
                    logger.debug(f"Step {step_counter}: List Available Tables")
                    logger.debug(f"   Tool: {tool_name}")

                elif 'schema' in tool_name.lower():
                    if not schema_phase:
                        logger.debug("SCHEMA INSPECTION PHASE")
                        schema_phase = True
                    logger.debug(f"Step {step_counter}: Check Table Structure")
                    logger.debug(f"   Tool: {tool_name}")
                    if 'table_names' in args:
                        logger.debug(f"   Target Tables: {args['table_names']}")

                elif 'query_checker' in tool_name.lower():
                    logger.debug(f"Step {step_counter}: Validate SQL Syntax")
                    logger.debug(f"   Tool: {tool_name}")
                    if 'query' in args:
                        sql_preview = args['query'][:100] + "..." if len(args['query']) > 100 else args['query']
                        logger.debug(f"   SQL Preview: {sql_preview}")

                elif 'sql_db_query' in tool_name.lower():
                    if not query_phase:
                        logger.debug("SQL GENERATION & EXECUTION PHASE")
                        query_phase = True
                    sql_query = args.get('query', '')
                    sql_attempts.append(sql_query)
                    attempt_num = len(sql_attempts)

                    if attempt_num > 1:
                        logger.debug(f"Step {step_counter}: Retry SQL Query (Attempt #{attempt_num})")
                    else:
                        logger.debug(f"Step {step_counter}: Execute SQL Query")

                    logger.debug(f"   Tool: {tool_name}")
                    logger.debug(f"   Generated SQL:")
                    logger.debug(f"   ┌{'─' * 56}┐")
                    # Format SQL for display (wrap if too long)
                    sql_lines = sql_query.split('\n') if '\n' in sql_query else [sql_query]
                    for line in sql_lines[:5]:  # Show first 5 lines max
                        display_line = line[:54] if len(line) > 54 else line
                        logger.debug(f"   │ {display_line:<54} │")
                    if len(sql_lines) > 5:
                        logger.debug(f"   │ {'... (truncated)':<54} │")
                    logger.debug(f"   └{'─' * 56}┘")

                    tool_call_times[step_counter] = time.time()

                else:
                    logger.debug(f"Step {step_counter}: {tool_name}")
                    if args:
                        logger.debug(f"   Arguments: {str(args)[:100]}")

        # Track ToolMessage (tool execution results)
        if msg_type == "ToolMessage" and hasattr(msg, 'name'):
            tool_name = msg.name
            content = msg.content

            # Calculate execution time if available
            if step_counter in tool_call_times:
                elapsed = time.time() - tool_call_times[step_counter]
                time_str = f" ({elapsed:.2f}s)"
            else:
                time_str = ""

            if 'list_tables' in tool_name.lower():
                # Extract table names from content
                tables = str(content).split(',') if ',' in str(content) else [str(content)]
                table_count = len(tables)
                logger.debug(f"   Result{time_str}: Found {table_count} tables")
                if table_count <= 15:
                    logger.debug(f"   Tables: {', '.join([t.strip() for t in tables[:15]])}")

            elif 'schema' in tool_name.lower():
                # Show schema info summary
                content_str = str(content)
                lines = content_str.split('\n')
                logger.debug(f"   Result{time_str}: Schema retrieved ({len(lines)} lines)")
                # Show first few columns
                column_lines = [l for l in lines if 'Column' in l or 'INTEGER' in l or 'VARCHAR' in l or 'TEXT' in l][:3]
                if column_lines:
                    logger.debug(f"   Sample Columns: {', '.join([l.strip()[:40] for l in column_lines])}")

            elif 'sql_db_query' in tool_name.lower():
                # Show query execution result
                if isinstance(content, list):
                    row_count = len(content)
                    logger.debug(f"   Result{time_str}: Query executed successfully")
                    logger.debug(f"   Rows returned: {row_count}")
                    logger.debug(f"   Data format: List[{type(content[0]).__name__ if content else 'Unknown'}]")
                elif isinstance(content, str):
                    if 'error' in content.lower() or 'exception' in content.lower():
                        logger.debug(f"   Error{time_str}: Query failed")
                        error_preview = content[:100] + "..." if len(content) > 100 else content
                        logger.debug(f"   Error message: {error_preview}")
                    else:
                        logger.debug(f"   Result{time_str}: Query executed")
                        logger.debug(f"   Response length: {len(content)} characters")
                else:
                    logger.debug(f"   Result{time_str}: Query executed")
                    logger.debug(f"   Response type: {type(content).__name__}")

            elif 'query_checker' in tool_name.lower():
                if 'error' in str(content).lower():
                    logger.debug(f"   Validation{time_str}: Issues found")
                else:
                    logger.debug(f"   Result{time_str}: SQL validated")

    # Summary
    logger.debug("=" * 60)
    logger.debug("Summary:")
    logger.debug(f"   Total steps: {step_counter}")
    if sql_attempts:
        logger.debug(f"   SQL attempts: {len(sql_attempts)}")
        if len(sql_attempts) > 1:
            logger.debug(f"   Note: Query was retried {len(sql_attempts) - 1} time(s)")
    logger.debug("=" * 60)
