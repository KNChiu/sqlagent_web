"""Data parsing utilities for SQL Agent

This module contains functions for:
- Extracting column names from SQL queries
- Extracting tool calls and results from agent messages
- Parsing query results into chart data format
"""

from typing import Optional, List, Dict, Any
import re
import json
import logging

logger = logging.getLogger(__name__)


# ============ SQL Extraction Functions ============

def extract_sql_from_tool_calls(messages: List[Any]) -> Optional[str]:
    """Extract SQL query from AIMessage tool_calls

    Searches for 'sql_db_query' tool calls in AIMessage objects
    and extracts the 'query' argument.

    Args:
        messages: List of agent messages (AIMessage, ToolMessage, etc.)

    Returns:
        SQL query string or None if not found
    """
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get('name', '') or ''
                if 'sql_db_query' in tool_name.lower():
                    args = tool_call.get('args', {})
                    if 'query' in args:
                        return args['query']
    return None


def extract_sql_from_content_fallback(messages: List[Any]) -> Optional[str]:
    """Extract SQL from message content using regex (fallback method)

    Scans message content in reverse order for SQL SELECT statements.

    Args:
        messages: List of agent messages

    Returns:
        Normalized SQL query string or None if not found
    """
    for msg in reversed(messages):
        if hasattr(msg, 'content'):
            content = str(msg.content)
            sql_match = re.search(
                r'(SELECT\s+.+?(?:FROM|;).*?)(?:\n|$)',
                content,
                re.IGNORECASE | re.DOTALL
            )
            if sql_match:
                sql_query = sql_match.group(1).strip()
                # Normalize whitespace
                sql_query = re.sub(r'\s+', ' ', sql_query)
                return sql_query
    return None


def extract_sql_from_messages(messages: List[Any]) -> Optional[str]:
    """Extract SQL query from agent messages (primary + fallback)

    Tries tool_calls first, falls back to content regex extraction.

    Args:
        messages: List of agent messages

    Returns:
        SQL query string or None
    """
    sql = extract_sql_from_tool_calls(messages)
    if sql:
        return sql

    # Fallback to content extraction
    return extract_sql_from_content_fallback(messages)


# ============ Result Extraction Functions ============

def parse_json_string(content: str) -> Optional[List]:
    """Parse JSON string to list

    Args:
        content: String content starting with '['

    Returns:
        Parsed list or None if parsing fails
    """
    try:
        result = json.loads(content)
        logger.info(f"Parsed as JSON (length: {len(result)}, "
                   f"first row type: {type(result[0]).__name__ if result else 'N/A'})")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        return None


def parse_literal_string(content: str) -> Optional[List]:
    """Parse Python literal string (using ast.literal_eval)

    Args:
        content: String content to parse

    Returns:
        Parsed list or None if parsing fails
    """
    try:
        import ast
        result = ast.literal_eval(content)
        logger.info(f"Parsed as Python literal (length: {len(result)})")
        return result
    except Exception as e:
        logger.warning(f"Literal eval failed: {e}")
        return None


def parse_string_content(content: str) -> Any:
    """Parse string content (JSON, literal, or plain text)

    Attempts multiple parsing strategies in order of likelihood.

    Args:
        content: String content from ToolMessage

    Returns:
        Parsed data (list, str, or None)
    """
    content_stripped = content.strip()

    # Strategy 1: JSON parsing
    if content_stripped.startswith('['):
        result = parse_json_string(content_stripped)
        if result is not None:
            return result

        # Strategy 2: Python literal eval (fallback)
        result = parse_literal_string(content_stripped)
        if result is not None:
            return result

        # Strategy 3: Keep as string if not SQL-related
        if 'SELECT' not in content.upper():
            logger.info("Keeping as string (fallback)")
            return content

    # Strategy 4: Plain text (no SQL)
    elif 'SELECT' not in content.upper():
        logger.info("Keeping as string (plain text)")
        return content

    return None


def parse_list_content(content: list) -> List:
    """Parse list content (already structured)

    Args:
        content: List content from ToolMessage

    Returns:
        Original list (no transformation needed)
    """
    logger.info(f"Parsed as list directly (length: {len(content)})")
    return content


def parse_tuple_content(content: tuple) -> List:
    """Parse tuple content (convert to list)

    Args:
        content: Tuple content from ToolMessage

    Returns:
        Converted list
    """
    result = list(content)
    logger.info(f"Converted tuple to list (length: {len(result)})")
    return result


def parse_tool_message_content(content: Any) -> Any:
    """Route content to appropriate parser based on type

    Central dispatcher for different content types.

    Args:
        content: ToolMessage content (various types)

    Returns:
        Parsed result (List[Dict], str, or None)
    """
    logger.info(f"Tool message content type: {type(content)}")
    logger.info(f"Content preview (first 200 chars): {str(content)[:200]}")

    if isinstance(content, list):
        return parse_list_content(content)
    elif isinstance(content, tuple):
        return parse_tuple_content(content)
    elif isinstance(content, str):
        return parse_string_content(content)
    else:
        logger.warning(f"Unexpected content type: {type(content)}")
        return None


# ============ Validation Functions ============

def validate_result_format(query_results: Any) -> bool:
    """Validate query results are in List[Dict] format

    Checks if results match expected QuerySQLDatabaseTool output format.

    Args:
        query_results: Parsed query results

    Returns:
        True if valid format, False otherwise
    """
    if not query_results or not isinstance(query_results, list):
        return False

    if len(query_results) == 0:
        return True  # Empty results are valid

    first_row = query_results[0]
    if not isinstance(first_row, dict):
        logger.warning(
            f"Unexpected query result format: expected List[Dict], "
            f"got List[{type(first_row).__name__}]"
        )
        return False

    return True


def extract_results_from_messages(messages: List[Any]) -> Optional[List[Dict]]:
    """Extract query results from ToolMessage objects

    Searches for 'sql_db_query' ToolMessages and extracts parsed results.

    Args:
        messages: List of agent messages

    Returns:
        Query results as List[Dict] or None if not found
    """
    for msg in messages:
        if not hasattr(msg, 'name'):
            continue

        msg_name = getattr(msg, 'name', '') or ''
        if 'sql_db_query' not in msg_name.lower():
            continue

        try:
            parsed_result = parse_tool_message_content(msg.content)
            if parsed_result is None:
                continue

            # Validate format
            if validate_result_format(parsed_result):
                return parsed_result
            else:
                # Return anyway but log warning (validation already logged)
                return parsed_result

        except Exception as e:
            logger.error(f"Unexpected error parsing tool results: {e}")
            continue

    return None


# ============ Facade Function (Backward Compatibility) ============

def extract_tool_calls_data(messages: List[Any]) -> tuple[Optional[str], Optional[List[Dict]]]:
    """Extract SQL query and results from tool calls in agent messages

    Facade function maintaining backward compatibility with existing callers.
    Delegates to specialized extraction functions for SQL and results.

    Args:
        messages: List of agent messages (AIMessage, ToolMessage, etc.)

    Returns:
        Tuple of (sql_query, query_results):
        - sql_query: SQL string or None
        - query_results: List[Dict] or None
    """
    sql_query = extract_sql_from_messages(messages)
    query_results = extract_results_from_messages(messages)

    if sql_query:
        logger.info(f"Extracted SQL: {sql_query[:150]}...")

    return sql_query, query_results


def parse_chart_data(query_result: Any, sql: str) -> Optional[Dict[str, Any]]:
    """Parse query results and generate chart data structure

    Handles both structured data (List[Dict]) and text-based results.
    """
    try:
        labels = []
        values = []

        # Case 1: Structured data (List of Dicts)
        if isinstance(query_result, list) and len(query_result) > 0:
            first_row = query_result[0]
            if isinstance(first_row, dict):
                keys = list(first_row.keys())
                if len(keys) >= 2:
                    # Assume first column is label, second is value
                    label_key = keys[0]
                    value_key = keys[1]

                    for row in query_result[:10]:  # Limit to 10 items
                        label = str(row.get(label_key, ''))
                        value = row.get(value_key)

                        # Try to convert value to float
                        try:
                            if isinstance(value, (int, float)):
                                labels.append(label)
                                values.append(float(value))
                            elif isinstance(value, str):
                                # Remove currency symbols and try to parse
                                clean_value = re.sub(r'[^\d.]', '', value)
                                if clean_value:
                                    labels.append(label)
                                    values.append(float(clean_value))
                        except (ValueError, TypeError):
                            continue

        # Case 2: Text-based result (fallback to original logic)
        elif isinstance(query_result, str):
            lines = query_result.strip().split('\n')

            for line in lines:
                if '|' in line or '\t' in line:
                    # Split and filter out empty parts
                    parts = [p.strip() for p in re.split(r'[|\t]+', line) if p.strip()]

                    # Skip header separator lines (e.g., |------|------|)
                    if len(parts) >= 2 and not parts[0].startswith('-'):
                        label_part = parts[0]
                        value_part = parts[-1]

                        # Try to extract numeric value (handle currency symbols like $)
                        value_match = re.search(r'\d+\.?\d*', value_part)
                        if value_match and label_part:
                            labels.append(label_part)
                            values.append(float(value_match.group()))

        # If we have data, create chart configuration
        if labels and values:
            # Determine chart type
            chart_type = 'bar' if any(keyword in sql.upper() for keyword in ['COUNT', 'SUM', 'AVG', 'TOTAL']) else 'line'

            return {
                'type': chart_type,
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'label': 'Result',
                        'data': values,
                        'backgroundColor': 'rgba(59, 130, 246, 0.5)',
                        'borderColor': 'rgb(59, 130, 246)',
                        'borderWidth': 2
                    }]
                }
            }

    except Exception as e:
        logger.warning(f"Chart parsing error: {e}")

    return None
