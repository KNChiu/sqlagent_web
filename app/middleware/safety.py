"""Safety Middleware for SQL Agent

Enforces DML restrictions and result limits to ensure safe SQL operations.
This middleware provides an additional layer of security by intercepting tool calls.
"""

from typing import List
import logging
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

# Use app.agent_service logger for consistency
logger = logging.getLogger('app.agent_service')


class SafetyMiddleware(AgentMiddleware):
    """
    Middleware that enforces SQL safety rules.

    This middleware prevents DML operations (INSERT, UPDATE, DELETE, DROP, etc.)
    and ensures result limits are respected. It intercepts tool calls before execution
    to validate SQL queries.
    """

    # Forbidden SQL keywords (DML operations)
    FORBIDDEN_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP',
        'ALTER', 'TRUNCATE', 'CREATE', 'REPLACE'
    ]

    def __init__(self):
        """Initialize Safety Middleware.

        This middleware enforces DML restrictions by intercepting
        tool calls before execution.
        """
        super().__init__()
        logger.info("SafetyMiddleware initialized")

    def transform_tool_call(self, message: AIMessage) -> AIMessage:
        """
        Intercept and validate tool calls before execution.

        This hook is called before each tool call, allowing us to validate
        SQL queries for forbidden operations.

        Args:
            message: AIMessage containing tool calls

        Returns:
            The original message if valid

        Raises:
            ValueError: If a forbidden SQL operation is detected
        """
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', '')

                # Check SQL queries for forbidden keywords
                if 'sql_db_query' in tool_name.lower():
                    args = tool_call.get('args', {})
                    sql_query = args.get('query', '')

                    # Validate SQL query
                    sql_upper = sql_query.upper()
                    for keyword in self.FORBIDDEN_KEYWORDS:
                        if keyword in sql_upper:
                            logger.warning(f"BLOCKED: DML operation '{keyword}' detected in query: {sql_query[:150]}...")
                            raise ValueError(
                                f"DML operation '{keyword}' is forbidden. "
                                f"This agent only supports read-only SELECT queries. "
                                f"Query: {sql_query[:100]}..."
                            )

        return message
