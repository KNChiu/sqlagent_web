"""Chart generation utilities with LLM enhancement

This module provides intelligent chart generation based on query results analysis.
Uses LLM to recommend optimal chart configurations with fallback to rule-based parsing.
"""

from typing import Optional, List, Dict, Any
import json
import re
import logging

logger = logging.getLogger(__name__)


class ChartBuilder:
    """Intelligent chart builder with LLM-enhanced data analysis

    This class encapsulates chart generation logic:
    - LLM-based chart type recommendation
    - Automatic data sorting and formatting
    - Fallback to rule-based parsing

    Supports both synchronous and asynchronous workflows.
    """

    # Chart color palette (Chart.js format with transparency)
    CHART_COLORS = [
        'rgba(59, 130, 246, 0.5)',   # Blue
        'rgba(16, 185, 129, 0.5)',   # Green
        'rgba(249, 115, 22, 0.5)',   # Orange
    ]

    # Maximum data points for chart clarity
    MAX_DATA_POINTS = 10

    # LLM confidence threshold (< 0.5 triggers fallback)
    MIN_CONFIDENCE = 0.5

    # Sample size for LLM analysis (reduce token usage)
    LLM_SAMPLE_SIZE = 3

    def __init__(self, llm):
        """Initialize ChartBuilder with LLM instance

        Args:
            llm: LangChain LLM instance for chart analysis
        """
        self.llm = llm

    def build_chart_data(
        self,
        user_query: str,
        sql_query: Optional[str],
        query_results: Any,
        response_text: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Build chart data with LLM enhancement fallback

        Primary method for generating chart data. Tries LLM-enhanced analysis
        first, falls back to rule-based parsing if enhancement fails.

        Args:
            user_query: User's natural language query
            sql_query: Generated SQL query string
            query_results: Query execution results (List[Dict] or string)
            response_text: Fallback text for parsing (process_query only)

        Returns:
            Chart.js compatible chart_data dict, or None if no visualization possible
        """
        chart_data = None

        # Try LLM-enhanced data analysis first
        if query_results and isinstance(query_results, list) and len(query_results) > 0:
            try:
                chart_data = self._enhance_data_with_llm(user_query, sql_query or "", query_results)
                if chart_data:
                    logger.info("Using LLM-enhanced chart")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")

        # Fallback to original parse_chart_data()
        if not chart_data:
            # Import here to avoid circular dependency
            from .parsers import parse_chart_data

            if query_results:
                chart_data = parse_chart_data(query_results, sql_query or "")
                logger.info("Using fallback chart parsing")
            elif response_text:  # Only used in process_query()
                chart_data = parse_chart_data(response_text, sql_query or "")

        return chart_data

    def _enhance_data_with_llm(
        self,
        user_query: str,
        sql_query: str,
        query_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze query results and recommend optimal chart configuration

        Args:
            user_query: Original user question in natural language
            sql_query: Executed SQL query
            query_results: Query results as List[Dict]

        Returns:
            Chart.js compatible chart_data dict, or None if analysis fails
        """
        try:
            # Step 1: Prepare sample data (first 3 rows to reduce token usage)
            sample_size = min(self.LLM_SAMPLE_SIZE, len(query_results))
            sample_data = query_results[:sample_size]
            column_names = list(sample_data[0].keys()) if sample_data else []

            # Step 2: Build LLM analysis prompt (imported from prompts.py)
            from .prompts import CHART_ANALYSIS_PROMPT

            prompt = CHART_ANALYSIS_PROMPT.format(
                user_query=user_query,
                sql_query=sql_query,
                columns=', '.join(column_names),
                sample_size=sample_size,
                sample_data=json.dumps(sample_data, ensure_ascii=False)
            )

            # Step 3: Call LLM (synchronous version)
            logger.info("Starting LLM data analysis...")
            response = self.llm.invoke([("user", prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Step 4: Parse JSON response (handle markdown code blocks)
            recommendation = self._parse_llm_response(response_text)
            if not recommendation:
                return None

            # Step 5: Validate confidence
            confidence = recommendation.get('confidence', 0)
            if confidence < self.MIN_CONFIDENCE:
                logger.warning(f"LLM confidence too low ({confidence}), using fallback")
                return None

            # Step 6: Build chart_data from recommendation
            chart_data = self._build_chart_from_recommendation(
                recommendation,
                query_results
            )

            logger.info(f"LLM-enhanced chart created (type: {recommendation['chart_type']}, confidence: {confidence:.2f})")
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

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response (handles markdown code blocks)

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try direct JSON parsing
            json_str = response_text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _build_chart_from_recommendation(
        self,
        recommendation: Dict[str, Any],
        query_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build Chart.js data structure from LLM recommendation

        Args:
            recommendation: LLM recommendation with chart_type, label_column, value_columns, etc.
            query_results: Full query results

        Returns:
            Chart.js compatible chart_data dict
        """
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

        # Limit to MAX_DATA_POINTS for chart clarity
        sorted_results = sorted_results[:self.MAX_DATA_POINTS]

        # Extract labels
        labels = [str(row.get(label_column, '')) for row in sorted_results]

        # Extract values (support multiple datasets)
        datasets = []
        for idx, value_col in enumerate(value_columns):
            values = self._extract_numeric_values(sorted_results, value_col)

            color = self.CHART_COLORS[idx % len(self.CHART_COLORS)]
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

        return chart_data

    def _extract_numeric_values(
        self,
        rows: List[Dict[str, Any]],
        value_column: str
    ) -> List[float]:
        """Extract numeric values from query results

        Handles type conversion and currency symbols.

        Args:
            rows: Query result rows
            value_column: Column name to extract

        Returns:
            List of numeric values (0 for non-numeric)
        """
        values = []
        for row in rows:
            value = row.get(value_column)

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

        return values
