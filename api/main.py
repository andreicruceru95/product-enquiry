"""
FastAPI application for the Product RAG system.
Provides REST API endpoints for conversational product search and quote generation.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from langchain_community.memory import ConversationBufferWindowMemory
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.product_rag_tools import create_product_rag_tools
from database.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User identifier")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session ID")
    message_id: int = Field(..., description="Message ID for feedback")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional response metadata")

class ProductSearchRequest(BaseModel):
    query: str = Field(..., description="Product search query")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional search filters")

class UserFeedback(BaseModel):
    session_id: str = Field(..., description="Session ID")
    message_id: int = Field(..., description="Message ID being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, description="Optional feedback comment")

class ProductComparisonRequest(BaseModel):
    product_ids: List[str] = Field(..., description="List of product IDs to compare")
    comparison_criteria: Optional[List[str]] = Field(None, description="Specific criteria to compare")

# Global variables for shared resources
session_manager: Optional[SessionManager] = None
agent_executor = None
rag_tools: List = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Starting Product RAG API...")
    
    # Pre-initialize dependencies
    await get_session_manager()
    await get_agent_executor()
    
    logger.info("Product RAG API ready!")
    
    yield
    
    # Shutdown
    global session_manager
    if session_manager:
        await session_manager.cleanup_old_sessions()
    logger.info("Product RAG API shutdown complete")

# FastAPI app initialization
app = FastAPI(
    title="Product RAG API",
    description="AI-powered product search and quote generation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent system message
SYSTEM_MESSAGE = """You are a helpful AI assistant specialized in product search and quote generation for an e-commerce platform.

When helping users, follow these guidelines:
1. For product searches, use embed_query_params and search_vector_store tools
2. Always rerank results using rerank_results for better relevance 
3. For quotes, gather product details and use create_quote tool
4. Use compute_cost tool for pricing calculations with bulk discounts
5. For similar products, use search_similar_items tool
6. Always format final responses using send_html_response tool
7. Be conversational and helpful while being precise

Think step by step about what the user needs:
- If they want product information or quotes, search and rerank results
- If they need pricing, use the cost computation tools
- Always end with a well-formatted HTML response"""

# Memory storage for conversations
conversation_memories: Dict[str, ConversationBufferWindowMemory] = {}

async def get_session_manager():
    """Dependency to get the session manager."""
    global session_manager
    if session_manager is None:
        session_manager = SessionManager("conversations.db")
        await session_manager.init_database()
    return session_manager

async def get_agent_executor():
    """Dependency to get the agent executor."""
    global agent_executor, rag_tools
    if agent_executor is None:
        # Initialize RAG tools
        rag_tools = create_product_rag_tools()
        
        # Initialize LLM (using ChatOllama for local deployment)
        llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
        
        # Create agent using LangGraph
        agent_executor = create_agent(llm, rag_tools, state_modifier=SYSTEM_MESSAGE)
        
        logger.info("Agent executor initialized with RAG tools")
    
    return agent_executor

def get_or_create_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Get or create conversation memory for a session."""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            memory_key="chat_history",
            return_messages=False
        )
    return conversation_memories[session_id]

# Lifespan events are now handled by the lifespan context manager above

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head>
            <title>Product RAG API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛒 Product RAG API</h1>
                <p>AI-powered product search and quote generation system</p>
                
                <h2>Available Endpoints:</h2>
                <div class="endpoint">
                    <strong>POST /chat/message</strong> - Send a message to the AI assistant
                </div>
                <div class="endpoint">
                    <strong>GET /search/products</strong> - Direct product search
                </div>
                <div class="endpoint">
                    <strong>POST /compare/products</strong> - Compare multiple products
                </div>
                <div class="endpoint">
                    <strong>POST /feedback</strong> - Submit user feedback
                </div>
                <div class="endpoint">
                    <strong>GET /analytics/weaknesses</strong> - View model performance analytics
                </div>
                
                <p><a href="/docs">📚 Interactive API Documentation</a></p>
            </div>
        </body>
    </html>
    """

@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(
    message: ChatMessage,
    background_tasks: BackgroundTasks,
    session_manager: SessionManager = Depends(get_session_manager),
    agent_executor = Depends(get_agent_executor)
):
    """
    Main chat endpoint for conversational product queries.
    Supports multi-turn conversations with session memory.
    """
    start_time = time.time()
    
    try:
        # Get or create session
        session_id = message.session_id
        if not session_id:
            session_id = await session_manager.create_conversation(user_id=message.user_id)
        
        # Get conversation memory
        memory = get_or_create_memory(session_id)
        
        # Add user message to session
        user_msg_id = await session_manager.add_message(
            session_id=session_id,
            role="user",
            content=message.message,
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # Prepare agent input
        agent_input = {
            "input": message.message,
            "chat_history": memory.buffer
        }
        
        # Execute agent
        agent_start = time.time()
        result = await asyncio.get_event_loop().run_in_executor(
            None, agent_executor.invoke, agent_input
        )
        agent_end = time.time()
        
        # Extract response
        response_content = result.get("output", "I apologize, but I couldn't process your request.")
        
        # Update memory
        memory.save_context(
            {"input": message.message},
            {"output": response_content}
        )
        
        # Add assistant message to session
        assistant_msg_id = await session_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=response_content,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "agent_steps": len(result.get("intermediate_steps", [])),
                "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])]
            },
            performance_metrics={
                "llm_time_ms": int((agent_end - agent_start) * 1000),
                "total_time_ms": int((time.time() - start_time) * 1000)
            }
        )
        
        # Background task: Add retrieval analytics
        background_tasks.add_task(
            add_retrieval_analytics,
            session_manager,
            assistant_msg_id,
            message.message,
            result.get("intermediate_steps", [])
        )
        
        return ChatResponse(
            response=response_content,
            session_id=session_id,
            message_id=assistant_msg_id,
            metadata={
                "response_time_ms": int((time.time() - start_time) * 1000),
                "tools_used": len(result.get("intermediate_steps", []))
            }
        )
        
    except Exception as e:
        logger.error(f"Chat message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/products")
async def search_products(
    query: str,
    limit: int = 10,
    agent_executor = Depends(get_agent_executor)
):
    """
    Direct product search endpoint without conversation context.
    """
    try:
        # Use the embedding and search tools directly
        embed_tool = None
        search_tool = None
        
        for tool in rag_tools:
            if tool.name == "embed_query_params":
                embed_tool = tool
            elif tool.name == "search_vector_store":
                search_tool = tool
        
        if not embed_tool or not search_tool:
            raise HTTPException(status_code=500, detail="Search tools not available")
        
        # Generate embeddings and search
        embeddings = embed_tool._run(query)
        results = search_tool._run(embeddings, limit)
        
        return {
            "query": query,
            "results": results,
            "count": len(results) if isinstance(results, list) else 0
        }
        
    except Exception as e:
        logger.error(f"Product search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare/products")
async def compare_products(
    comparison: ProductComparisonRequest,
    agent_executor = Depends(get_agent_executor)
):
    """
    Compare multiple products by their IDs.
    """
    try:
        # Find search_item_by_index tool
        search_by_index_tool = None
        for tool in rag_tools:
            if tool.name == "search_item_by_index":
                search_by_index_tool = tool
                break
        
        if not search_by_index_tool:
            raise HTTPException(status_code=500, detail="Product lookup tool not available")
        
        # Get product details for each ID
        products = []
        for product_id in comparison.product_ids:
            product_data = search_by_index_tool._run(product_id)
            if product_data.get("found", False):
                products.append(product_data)
        
        return {
            "comparison_request": comparison.dict(),
            "products": products,
            "count": len(products)
        }
        
    except Exception as e:
        logger.error(f"Product comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(
    feedback: UserFeedback,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Submit user feedback for model improvement.
    """
    try:
        await session_manager.add_user_feedback(
            session_id=feedback.session_id,
            message_id=feedback.message_id,
            feedback_type="rating",
            feedback_value=str(feedback.rating),
            metadata={"comment": feedback.comment} if feedback.comment else None
        )
        
        return {"message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/weaknesses")
async def get_model_weaknesses(
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get model weakness analysis for improvement.
    """
    try:
        weaknesses = await session_manager.get_model_weaknesses()
        return weaknesses
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/stats")
async def get_session_stats(
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get overall session and usage statistics.
    """
    try:
        stats = await session_manager.get_session_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Background task functions
async def add_retrieval_analytics(
    session_manager: SessionManager,
    message_id: int,
    query: str,
    intermediate_steps: List
):
    """
    Background task to add retrieval analytics.
    """
    try:
        # Extract retrieval information from agent steps
        retrieval_data = {
            "product_ids": [],
            "embedding_scores": [],
            "reranker_scores": [],
            "final_ranking": []
        }
        
        query_intent = "general"  # Simple intent classification
        if any(word in query.lower() for word in ["quote", "price", "cost"]):
            query_intent = "quote_request"
        elif any(word in query.lower() for word in ["compare", "vs", "versus"]):
            query_intent = "comparison"
        elif any(word in query.lower() for word in ["similar", "like", "recommend"]):
            query_intent = "recommendation"
        else:
            query_intent = "product_search"
        
        # Extract data from intermediate steps
        for step in intermediate_steps:
            action, observation = step
            if action.tool == "search_vector_store" and isinstance(observation, list):
                for result in observation:
                    if isinstance(result, dict) and "id" in result:
                        retrieval_data["product_ids"].append(result["id"])
                        retrieval_data["embedding_scores"].append(result.get("similarity_score", 0))
        
        await session_manager.add_retrieval_analytics(
            message_id=message_id,
            query=query,
            query_intent=query_intent,
            retrieval_results=retrieval_data
        )
        
    except Exception as e:
        logger.error(f"Error adding retrieval analytics: {e}")

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )