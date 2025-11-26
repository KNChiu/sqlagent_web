"""API Routes for SQL Agent

This module defines all API endpoints:
- POST /api/query: Synchronous query processing
- POST /api/query/stream: Streaming query with SSE
- GET /: Serve frontend HTML
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import time
import logging

from .agent_service import SQLAgentService

logger = logging.getLogger(__name__)


# Create router
router = APIRouter()

# Initialize SQL Agent Service with lazy initialization (dependency injection)
_agent_service: Optional[SQLAgentService] = None


def get_agent_service() -> SQLAgentService:
    """Lazy initialization of SQL Agent Service - only creates connection on first request"""
    global _agent_service
    if _agent_service is None:
        logger.info("Initializing SQL Agent Service (first request)...")
        _agent_service = SQLAgentService()
    return _agent_service


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = []


class QueryResponse(BaseModel):
    message: str
    sql: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None
    raw_data: Optional[List[Dict[str, Any]]] = None


@router.post("/api/query", response_model=QueryResponse)
async def query_database(
    request: QueryRequest,
    agent_service: SQLAgentService = Depends(get_agent_service)
):
    """Process natural language query and return SQL results with chart data"""
    start_time = time.time()
    logger.info(f"Received query: '{request.query}' (conversation history: {len(request.conversation_history)} messages)")

    try:
        result = agent_service.process_query(
            query=request.query,
            conversation_history=request.conversation_history
        )

        duration = time.time() - start_time
        logger.info(
            f"Query completed in {duration:.2f}s "
            f"(SQL: {'Yes' if result['sql'] else 'No'}, "
            f"Chart: {'Yes' if result['chart_data'] else 'No'})"
        )

        return QueryResponse(
            message=result["message"],
            sql=result["sql"],
            chart_data=result["chart_data"],
            raw_data=result["raw_data"]
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Query processing error after {duration:.2f}s: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@router.post("/api/query/stream")
async def query_database_stream(
    request: QueryRequest,
    agent_service: SQLAgentService = Depends(get_agent_service)
):
    """Stream agent execution with real-time status updates using Server-Sent Events"""
    logger.info(f"Starting streaming query: '{request.query}'")

    async def event_generator():
        try:
            async for event_data in agent_service.stream_query(
                query=request.query,
                conversation_history=request.conversation_history
            ):
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

            logger.info("Streaming query completed successfully")

        except Exception as e:
            logger.error(f"Streaming query error: {type(e).__name__}: {str(e)}")
            # Send error event
            error_event = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )


@router.get("/")
async def read_index():
    """Serve the frontend HTML page"""
    return FileResponse("index.html")
