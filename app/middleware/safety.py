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

    def __init__(self, query_limit: int = 5):
        """
        Initialize Safety Middleware.

        Args:
            query_limit: Default result limit for SQL queries
        """
        super().__init__()
        self.query_limit = query_limit
        self.intercepted_count = 0
        logger.info(f"SafetyMiddleware initialized (query_limit={query_limit})")

    @property
    def system_prompt(self) -> str:
        """System prompt with safety instructions"""
        return f"""
SAFETY RULES (CRITICAL):
1. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
2. Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {self.query_limit} results.
3. However, if the user explicitly requests a different number (e.g., "show me 10 rows", "give me 20 records"), you MUST respect their request and override the default limit.

EFFICIENCY RULES (CRITICAL):
1. Once you execute a query and get valid results, STOP immediately and provide your answer.
2. Do NOT rewrite queries just to change output format, column names, or data presentation.
3. Do NOT execute the same query multiple times with minor variations.
4. If you get data back from sql_db_query, that means the query was successful - answer the user's question directly.
5. Only retry if you get an ERROR message. Valid results (even if formatted differently than expected) should be used immediately.
"""

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
                            self.intercepted_count += 1
                            logger.warning(f"BLOCKED: DML operation '{keyword}' detected in query: {sql_query[:150]}...")
                            raise ValueError(
                                f"DML operation '{keyword}' is forbidden. "
                                f"This agent only supports read-only SELECT queries. "
                                f"Query: {sql_query[:100]}..."
                            )

        return message
