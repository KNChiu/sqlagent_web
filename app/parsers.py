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


def extract_column_names_from_sql(sql: str) -> List[str]:
    """Extract column names from SQL SELECT statement"""
    if not sql:
        return []

    try:
        # Remove comments and extra whitespace
        sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        sql_clean = ' '.join(sql_clean.split())

        # Extract SELECT ... FROM portion
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_clean, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return []

        select_clause = select_match.group(1).strip()

        # Split by comma (but not commas inside functions)
        columns = []
        paren_depth = 0
        current_col = []

        for char in select_clause:
            if char == '(':
                paren_depth += 1
                current_col.append(char)
            elif char == ')':
                paren_depth -= 1
                current_col.append(char)
            elif char == ',' and paren_depth == 0:
                columns.append(''.join(current_col).strip())
                current_col = []
            else:
                current_col.append(char)

        # Add last column
        if current_col:
            columns.append(''.join(current_col).strip())

        # Extract alias names (AS alias_name) or use full expression
        column_names = []
        for col in columns:
            # Look for AS alias
            as_match = re.search(r'\bAS\s+([^\s,]+)', col, re.IGNORECASE)
            if as_match:
                column_names.append(as_match.group(1).strip())
            else:
                # Use the last word (might be column name without AS)
                parts = col.split()
                if parts:
                    # Remove quotes if present
                    col_name = parts[-1].strip('"\'')
                    column_names.append(col_name)

        return column_names
    except Exception as e:
        logger.warning(f"Error extracting column names: {e}")
        return []


def extract_tool_calls_data(messages: List[Any]) -> tuple[Optional[str], Optional[List[Dict]]]:
    """Extract SQL query and results from tool calls in agent messages"""
    sql_query = None
    query_results = None

    for msg in messages:
        # Check for tool calls in AIMessage
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                # Extract SQL from sql_db_query tool
                tool_name = tool_call.get('name', '') or ''
                if 'sql_db_query' in tool_name.lower():
                    args = tool_call.get('args', {})
                    if 'query' in args:
                        sql_query = args['query']

        # Check for tool results in ToolMessage
        if hasattr(msg, 'name'):
            msg_name = getattr(msg, 'name', '') or ''
            if 'sql_db_query' in msg_name.lower():
                content = msg.content

                # Log tool message content
                logger.debug(f"Tool message content type: {type(content)}")
                logger.debug(f"Content preview (first 500 chars): {str(content)[:500]}")

                # Try to parse structured data from tool response
                try:
                    # Case 1: Content is already a list (structured data)
                    if isinstance(content, list):
                        query_results = content
                        logger.debug("Parsed as list directly")
                    # Case 2: Content is a tuple (convert to list)
                    elif isinstance(content, tuple):
                        query_results = list(content)
                        logger.debug("Converted tuple to list")
                    # Case 3: Content is JSON string
                    elif isinstance(content, str):
                        content_stripped = content.strip()

                        # Try JSON parsing
                        if content_stripped.startswith('['):
                            try:
                                query_results = json.loads(content_stripped)
                                logger.debug("Parsed as JSON")
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON parse failed: {e}")
                                # Try Python literal eval
                                try:
                                    import ast
                                    query_results = ast.literal_eval(content_stripped)
                                    logger.debug("Parsed as Python literal")
                                except Exception as e2:
                                    logger.debug(f"Literal eval failed: {e2}")
                                    # Keep as string for text-based parsing
                                    if 'SELECT' not in content.upper():
                                        query_results = content
                                        logger.debug("Keeping as string (fallback)")
                        else:
                            # Plain text format
                            if 'SELECT' not in content.upper():
                                query_results = content
                                logger.debug("Keeping as string (plain text)")
                except Exception as e:
                    logger.debug(f"Unexpected error parsing tool results: {e}")
                    pass

    # Fallback: extract SQL from message content using regex
    if not sql_query:
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                content = str(msg.content)
                sql_match = re.search(r'(SELECT\s+.+?(?:FROM|;).*?)(?:\n|$)', content, re.IGNORECASE | re.DOTALL)
                if sql_match:
                    sql_query = sql_match.group(1).strip()
                    sql_query = re.sub(r'\s+', ' ', sql_query)
                    break

    # Convert List[Tuple] to List[Dict] if needed
    if query_results and isinstance(query_results, list) and len(query_results) > 0:
        first_row = query_results[0]
        # If the result is a list of tuples, convert to list of dicts
        if isinstance(first_row, tuple):
            # Extract column names from SQL query
            column_names = extract_column_names_from_sql(sql_query)
            if column_names and len(column_names) == len(first_row):
                # Convert tuples to dicts
                query_results = [
                    dict(zip(column_names, row))
                    for row in query_results
                ]
                logger.debug(f"Converted List[Tuple] to List[Dict] with columns: {column_names}")
            else:
                # Fallback: use generic column names
                column_names = [f"Column_{i+1}" for i in range(len(first_row))]
                query_results = [
                    dict(zip(column_names, row))
                    for row in query_results
                ]
                logger.debug(f"Using generic column names: {column_names}")

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
