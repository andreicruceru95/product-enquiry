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

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

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

class UserAnalyticsQuery(BaseModel):
    user_id: str = Field(..., description="User ID to analyze")
    analysis_question: str = Field(..., description="Specific question about the user's behavior")
    include_sessions: Optional[List[str]] = Field(None, description="Specific session IDs to analyze")
    date_range_days: Optional[int] = Field(30, description="Number of days to look back")
    max_messages_per_session: Optional[int] = Field(50, description="Limit messages per session for token management")

class UserAnalyticsResponse(BaseModel):
    user_id: str
    analysis_summary: str
    statistics: Dict[str, Any]
    insights: List[str]
    conversation_topics: List[str]
    sessions_analyzed: int
    total_messages: int
    analysis_timestamp: str

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
SYSTEM_MESSAGE = """You are a helpful AI assistant specialized in product search, quote generation, and mathematical analysis for an e-commerce platform.

When helping users, follow these guidelines:
1. For product searches, use embed_query_params and search_vector_store tools
2. Always rerank results using rerank_results for better relevance 
3. For quotes, gather product details and use create_quote tool
4. Use compute_cost tool for pricing calculations with bulk discounts
5. For similar products, use search_similar_items tool
6. For mathematical analysis, use statistical_analysis, price_comparison, and financial_calculator tools
7. Always format final responses using send_html_response tool
8. Be conversational and helpful while being precise

Available Mathematical Tools:
- statistical_analysis: Calculate mean, median, std dev, percentiles for price analysis
- price_comparison: Compare prices across products, find best deals and savings
- financial_calculator: Calculate interest, discounts, tax, ROI, loan payments

Think step by step about what the user needs:
- If they want product information or quotes, search and rerank results
- If they need statistical analysis of prices, use statistical_analysis tool
- If they want to compare prices or find best deals, use price_comparison tool
- If they need financial calculations, use financial_calculator tool
- Always end with a well-formatted HTML response"""

# Memory storage for conversations
conversation_memories: Dict[str, List[BaseMessage]] = {}

async def get_session_manager():
    """Dependency to get the session manager."""
    global session_manager
    if session_manager is None:
        session_manager = SessionManager("conversations.db")
        await session_manager.init_database()
    return session_manager

class SimpleRAGAgent:
    """Simple RAG agent that uses tools directly without LangChain agents framework."""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        logger.info(f"Initialized SimpleRAGAgent with tools: {list(self.tools.keys())}")
        
    def invoke(self, inputs):
        """Process user input and execute appropriate tools."""
        user_message = inputs["input"]
        chat_history = inputs.get("chat_history", "")
        
        # Create a simple prompt that includes tool descriptions
        tool_descriptions = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
        
        prompt = f"""{SYSTEM_MESSAGE}

Available tools:
{tool_descriptions}

Chat History:
{chat_history}

User Query: {user_message}

Please help the user with their query. You can use the available tools by mentioning them in your response."""
        
        # Get LLM response
        response = self.llm.invoke(prompt)
        
        # Simple tool execution based on keywords in user message
        intermediate_steps = []
        
        # Basic tool routing based on keywords
        if any(word in user_message.lower() for word in ["search", "find", "product", "laptop", "phone", "headphone", "wireless", "cost", "price"]):
            try:
                # Use embed and search tools in sequence
                embed_tool = self.tools.get("embed_query_params")
                search_tool = self.tools.get("search_vector_store")
                
                if embed_tool and search_tool:
                    logger.info(f"Found tools - embed_tool: {type(embed_tool)}, search_tool: {type(search_tool)}")
                    # First embed the query
                    embeddings = embed_tool._run(user_message)
                    intermediate_steps.append(("embed_query_params", "Query embedded"))
                    
                    # Then search the vector store
                    search_results = search_tool._run(embeddings, k=5)
                    intermediate_steps.append(("search_vector_store", f"Found {len(search_results) if isinstance(search_results, list) else 0} results"))
                    
                    # Format the search results for the LLM
                    formatted_results = ""
                    if isinstance(search_results, list) and search_results:
                        for i, result in enumerate(search_results[:3], 1):
                            if isinstance(result, dict):
                                metadata = result.get("metadata", {})
                                title = metadata.get("title", "Unknown Product")
                                price = metadata.get("price", 0)
                                category = metadata.get("main_category", "")
                                formatted_results += f"{i}. {title} - ${price} ({category})\n"
                    
                    # Update the response with search results
                    enhanced_prompt = f"""{prompt}

Product Search Results:
{formatted_results if formatted_results else "No products found matching your query."}

Based on the search results above, please provide a helpful response to the user's query about {user_message}."""
                    
                    response = self.llm.invoke(enhanced_prompt)
                    
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                intermediate_steps.append(("error", str(e)))
        
        # Extract content from response object
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "output": response_content,
            "intermediate_steps": intermediate_steps
        }

async def get_agent_executor():
    """Dependency to get the agent executor."""
    global agent_executor, rag_tools
    if agent_executor is None:
        # Initialize RAG tools
        rag_tools = create_product_rag_tools()
        
        # Initialize LLM (using ChatOllama for local deployment)
        llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
        
        # Create simple agent
        agent_executor = SimpleRAGAgent(llm, rag_tools)
        
        logger.info("Simple RAG agent initialized with tools")
    
    return agent_executor

def get_or_create_memory(session_id: str) -> List[BaseMessage]:
    """Get or create conversation memory for a session."""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = []
    # Keep only last 20 messages (10 exchanges)
    if len(conversation_memories[session_id]) > 20:
        conversation_memories[session_id] = conversation_memories[session_id][-20:]
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
            # Generate new session ID
            import uuid
            session_id = str(uuid.uuid4())
        
        # Check if session exists, create if not
        try:
            existing_session = await session_manager.get_conversation(session_id)
        except:
            existing_session = None
        
        if not existing_session:
            session_id = await session_manager.create_conversation(
                session_id=session_id, 
                user_id=message.user_id
            )
            logger.info(f"Created new session: {session_id} for user: {message.user_id}")
        
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
        chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in memory[-10:]])
        agent_input = {
            "input": message.message,
            "chat_history": chat_history
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
        memory.append(HumanMessage(content=message.message))
        memory.append(AIMessage(content=response_content))
        
        # Add assistant message to session
        assistant_msg_id = await session_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=response_content,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "agent_steps": len(result.get("intermediate_steps", [])),
                "tools_used": [step[0] for step in result.get("intermediate_steps", []) if isinstance(step, tuple)]
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

@app.post("/sessions/create")
async def create_session(
    user_id: str,
    session_id: Optional[str] = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Create a new chat session."""
    try:
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        
        created_session_id = await session_manager.create_conversation(
            session_id=session_id,
            user_id=user_id
        )
        
        return {
            "session_id": created_session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get session information and message history."""
    try:
        session = await session_manager.get_conversation(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        messages = await session_manager.get_conversation_messages(session_id)
        return {
            "session_id": session_id,
            "user_id": session.get("user_id"),
            "created_at": session.get("created_at"),
            "message_count": len(messages) if messages else 0,
            "messages": messages or []
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Delete a session and its message history."""
    try:
        # Delete the session and all associated messages
        deleted = await session_manager.delete_conversation(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {"message": f"Session {session_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/user", response_model=UserAnalyticsResponse)
async def analyze_user_conversations(
    query: UserAnalyticsQuery,
    background_tasks: BackgroundTasks,
    session_manager: SessionManager = Depends(get_session_manager),
    agent_executor = Depends(get_agent_executor)
):
    """
    Analyze a user's conversation patterns across multiple sessions.
    Provides insights on topics, interests, conversation patterns, and more.
    
    Example queries:
    - "What are this user's main product interests?"
    - "How engaged is this user with the platform?"
    - "What's this user's typical shopping behavior?"
    - "Does this user prefer budget or premium products?"
    """
    start_time = time.time()
    
    try:
        # Get user's conversations with date filtering
        user_sessions = await get_user_sessions_with_messages(
            session_manager, 
            query.user_id, 
            query.date_range_days or 30,
            query.max_messages_per_session or 50
        )
        
        if not user_sessions:
            raise HTTPException(
                status_code=404, 
                detail=f"No conversations found for user {query.user_id} in the last {query.date_range_days or 30} days"
            )
        
        # Filter specific sessions if requested
        if query.include_sessions:
            user_sessions = [s for s in user_sessions if s['session_id'] in query.include_sessions]
            if not user_sessions:
                raise HTTPException(
                    status_code=404,
                    detail=f"None of the specified sessions found for user {query.user_id}"
                )
        
        # Calculate basic statistics
        stats = calculate_user_statistics(user_sessions)
        
        # Prepare conversation data for LLM analysis (token-managed)
        conversation_summary = prepare_conversation_summary_for_llm(
            user_sessions, 
            max_tokens=3000  # Leave room for analysis prompt and response
        )
        
        # Create analysis prompt
        analysis_prompt = create_user_analysis_prompt(
            query.analysis_question,
            conversation_summary,
            stats
        )
        
        # Get LLM analysis
        agent_input = {
            "input": analysis_prompt,
            "chat_history": ""
        }
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, agent_executor.invoke, agent_input
        )
        
        analysis_result = result.get("output", "Analysis could not be completed.")
        
        # Extract insights and topics from the analysis
        insights, topics = extract_insights_and_topics(analysis_result, user_sessions)
        
        response = UserAnalyticsResponse(
            user_id=query.user_id,
            analysis_summary=analysis_result,
            statistics=stats,
            insights=insights,
            conversation_topics=topics,
            sessions_analyzed=len(user_sessions),
            total_messages=sum(len(session['messages']) for session in user_sessions),
            analysis_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"User analytics completed for {query.user_id}: {len(user_sessions)} sessions, {response.total_messages} messages")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/user", response_model=UserAnalyticsResponse)
async def analyze_user_conversations(
    query: UserAnalyticsQuery,
    background_tasks: BackgroundTasks,
    session_manager: SessionManager = Depends(get_session_manager),
    agent_executor = Depends(get_agent_executor)
):
    """
    Analyze a user's conversation patterns across multiple sessions.
    Provides insights on topics, interests, conversation patterns, and more.
    """
    start_time = time.time()
    
    try:
        # Get user's conversations with date filtering
        user_sessions = await get_user_sessions_with_messages(
            session_manager, 
            query.user_id, 
            query.date_range_days,
            query.max_messages_per_session
        )
        
        if not user_sessions:
            raise HTTPException(
                status_code=404, 
                detail=f"No conversations found for user {query.user_id}"
            )
        
        # Calculate basic statistics
        stats = calculate_user_statistics(user_sessions)
        
        # Prepare conversation data for LLM analysis
        conversation_summary = prepare_conversation_summary_for_llm(
            user_sessions, 
            max_tokens=3000  # Leave room for analysis prompt
        )
        
        # Create analysis prompt
        analysis_prompt = create_user_analysis_prompt(
            query.analysis_question,
            conversation_summary,
            stats
        )
        
        # Get LLM analysis
        agent_input = {
            "input": analysis_prompt,
            "chat_history": ""
        }
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, agent_executor.invoke, agent_input
        )
        
        analysis_result = result.get("output", "Analysis could not be completed.")
        
        # Extract insights and topics from the analysis
        insights, topics = extract_insights_and_topics(analysis_result, user_sessions)
        
        return UserAnalyticsResponse(
            user_id=query.user_id,
            analysis_summary=analysis_result,
            statistics=stats,
            insights=insights,
            conversation_topics=topics,
            sessions_analyzed=len(user_sessions),
            total_messages=sum(len(session['messages']) for session in user_sessions),
            analysis_timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User analytics error: {e}")
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
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                # action is the tool name string, not an object with .tool attribute
                if action == "search_vector_store" and isinstance(observation, str):
                    # For our SimpleRAGAgent, observation is a string description
                    # We can extract product count info if needed
                    pass
                elif hasattr(action, 'tool') and action.tool == "search_vector_store" and isinstance(observation, list):
                    # This would be for proper LangChain agent format
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

# User Analytics Helper Functions
async def get_user_sessions_with_messages(
    session_manager: SessionManager, 
    user_id: str, 
    days_back: int = 30,
    max_messages_per_session: int = 50
) -> List[Dict[str, Any]]:
    """Get user sessions with messages for analysis."""
    import aiosqlite
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    async with aiosqlite.connect(session_manager.db_path) as db:
        # Get user sessions within date range
        async with db.execute("""
            SELECT session_id, started_at, last_activity, metadata
            FROM conversations 
            WHERE user_id = ? AND started_at >= ?
            ORDER BY started_at DESC
        """, (user_id, cutoff_date.isoformat())) as cursor:
            sessions = await cursor.fetchall()
        
        user_sessions = []
        for session in sessions:
            session_id, started_at, last_activity, metadata = session
            
            # Get messages for this session (limited) - use table prefixes to avoid ambiguity
            async with db.execute("""
                SELECT m.role, m.content, m.timestamp, m.metadata
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.session_id = ?
                ORDER BY m.timestamp ASC
                LIMIT ?
            """, (session_id, max_messages_per_session)) as cursor:
                messages = await cursor.fetchall()
            
            message_list = []
            for msg in messages:
                message_list.append({
                    'role': msg[0],
                    'content': msg[1],
                    'timestamp': msg[2],
                    'metadata': json.loads(msg[3]) if msg[3] else {}
                })
            
            user_sessions.append({
                'session_id': session_id,
                'started_at': started_at,
                'last_activity': last_activity,
                'metadata': json.loads(metadata) if metadata else {},
                'messages': message_list
            })
    
    return user_sessions

def calculate_user_statistics(user_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate basic statistics from user sessions."""
    total_messages = sum(len(session['messages']) for session in user_sessions)
    user_messages = sum(1 for session in user_sessions for msg in session['messages'] if msg['role'] == 'user')
    assistant_messages = sum(1 for session in user_sessions for msg in session['messages'] if msg['role'] == 'assistant')
    
    # Calculate session durations (if we have timestamps)
    session_durations = []
    for session in user_sessions:
        if session['messages'] and len(session['messages']) > 1:
            first_msg = session['messages'][0]['timestamp']
            last_msg = session['messages'][-1]['timestamp']
            try:
                duration = (datetime.fromisoformat(last_msg) - datetime.fromisoformat(first_msg)).total_seconds() / 60
                session_durations.append(duration)
            except:
                pass
    
    # Extract common keywords from user messages
    user_content = ' '.join([msg['content'].lower() for session in user_sessions 
                            for msg in session['messages'] if msg['role'] == 'user'])
    
    # Simple keyword frequency (you could enhance this with NLP)
    common_words = ['laptop', 'phone', 'headphone', 'price', 'cost', 'cheap', 'expensive', 'wireless', 'gaming']
    keyword_counts = {word: user_content.count(word) for word in common_words if user_content.count(word) > 0}
    
    return {
        'total_sessions': len(user_sessions),
        'total_messages': total_messages,
        'user_messages': user_messages,
        'assistant_messages': assistant_messages,
        'avg_messages_per_session': total_messages / len(user_sessions) if user_sessions else 0,
        'avg_session_duration_minutes': sum(session_durations) / len(session_durations) if session_durations else 0,
        'keyword_frequencies': keyword_counts,
        'date_range': {
            'earliest': min([s['started_at'] for s in user_sessions]) if user_sessions else None,
            'latest': max([s['last_activity'] for s in user_sessions]) if user_sessions else None
        }
    }

def prepare_conversation_summary_for_llm(user_sessions: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
    """Prepare conversation summary with token management."""
    summary_parts = []
    estimated_tokens = 0
    
    for i, session in enumerate(user_sessions):
        session_summary = f"Session {i+1} ({session['started_at'][:10]}):\n"
        
        # Add key user messages (sample to stay within token limits)
        user_messages = [msg for msg in session['messages'] if msg['role'] == 'user']
        
        # Take every nth message if too many
        if len(user_messages) > 10:
            step = len(user_messages) // 10
            user_messages = user_messages[::step][:10]
        
        for msg in user_messages:
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            session_summary += f"  User: {content}\n"
        
        # Rough token estimation (4 chars per token)
        session_tokens = len(session_summary) // 4
        
        if estimated_tokens + session_tokens > max_tokens:
            break
        
        summary_parts.append(session_summary)
        estimated_tokens += session_tokens
    
    return "\n".join(summary_parts)

def create_user_analysis_prompt(question: str, conversation_summary: str, stats: Dict[str, Any]) -> str:
    """Create analysis prompt for the LLM."""
    return f"""Analyze this user's conversation patterns and behavior based on the following data:

USER QUESTION: {question}

CONVERSATION STATISTICS:
- Total Sessions: {stats['total_sessions']}
- Total Messages: {stats['total_messages']} (User: {stats['user_messages']}, Assistant: {stats['assistant_messages']})
- Average Messages per Session: {stats['avg_messages_per_session']:.1f}
- Average Session Duration: {stats['avg_session_duration_minutes']:.1f} minutes
- Top Keywords: {stats['keyword_frequencies']}
- Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}

CONVERSATION SAMPLES:
{conversation_summary}

Please provide a comprehensive analysis addressing the user's question. Focus on:
1. User behavior patterns
2. Product interests and preferences  
3. Shopping habits and tendencies
4. Communication style and engagement
5. Specific insights related to the question asked

Be specific and provide actionable insights based on the conversation data."""

def extract_insights_and_topics(analysis_result: str, user_sessions: List[Dict[str, Any]]) -> tuple:
    """Extract structured insights and topics from LLM analysis."""
    # Extract insights (look for bullet points or numbered lists)
    insights = []
    lines = analysis_result.split('\n')
    for line in lines:
        line = line.strip()
        if (line.startswith('- ') or line.startswith('• ') or 
            any(line.startswith(f'{i}.') for i in range(1, 10))):
            insights.append(line.lstrip('- •0123456789. '))
    
    # Extract topics from conversation content
    all_content = ' '.join([msg['content'].lower() for session in user_sessions 
                           for msg in session['messages'] if msg['role'] == 'user'])
    
    # Simple topic extraction (could be enhanced with NLP)
    topic_keywords = {
        'laptops': ['laptop', 'computer', 'pc', 'macbook'],
        'phones': ['phone', 'smartphone', 'iphone', 'android'],
        'audio': ['headphone', 'earbuds', 'speaker', 'audio'],
        'gaming': ['gaming', 'game', 'controller', 'console'],
        'accessories': ['mouse', 'keyboard', 'cable', 'charger'],
        'pricing': ['price', 'cost', 'cheap', 'expensive', 'budget', 'deal']
    }
    
    topics = []
    for topic, keywords in topic_keywords.items():
        if any(keyword in all_content for keyword in keywords):
            topics.append(topic)
    
    # Fallback insights if none found
    if not insights:
        insights = [
            f"User has {len(user_sessions)} conversation sessions",
            f"Shows interest in {', '.join(topics) if topics else 'various products'}",
            "Regular engagement with the platform"
        ]
    
    return insights[:5], topics  # Limit to top 5 insights

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )