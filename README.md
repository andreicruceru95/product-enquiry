# Product RAG Application

## Overview

AI-powered product search and quote generation system using RAG (Retrieval-Augmented Generation) with LLM orchestration. Built with LangChain, ChromaDB, and FastAPI.

## Features

- **Conversational Product Search**: Natural language queries for product information
- **Quote Generation**: Automated HTML quote generation with pricing calculations
- **Multi-turn Conversations**: Session-based memory for context-aware responses
- **Advanced Caching**: Multi-level caching with Redis and DiskCache
- **Model Analytics**: Conversation analysis for model improvement
- **CUDA Acceleration**: GPU support for faster embeddings and inference

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **LLM Integration**: LangChain with Ollama (local) or cloud models
- **Vector DB**: ChromaDB with BGE embeddings
- **Embeddings**: BAAI/bge-large-en-v1.5 (replacing OpenAI)
- **Reranking**: BAAI/bge-reranker-base
- **Caching**: Redis + DiskCache
- **Session Storage**: SQLite with aiosqlite
- **Templates**: Jinja2 for HTML responses

## Quick Start

### 1. Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.6+ (optional, falls back to CPU)
- Redis server (optional, uses disk cache fallback)

### 2. Installation & Setup

```bash
# Clone and setup
git clone <your-repo>
cd Product-RAG

# Run complete setup (creates venv, installs deps, initializes DBs)
python startup.py

# Or manual setup:
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Initialize Databases

```bash
# Build embeddings database (will take 1-2 hours for full dataset)
python database_builder.py

# Or limit for testing
python startup.py --max-products 1000
```

### 4. Start the API

```bash
# Using startup script (recommended)
python startup.py

# Or directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the API

Visit `http://localhost:8000` for the web interface or `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Main Chat Interface
```bash
POST /chat/message
{
    "message": "I need a quote for 10 Dell laptops and 20 wireless mice",
    "session_id": "optional-session-id",
    "user_id": "optional-user-id"
}
```

### Direct Product Search
```bash
GET /search/products?query=Dell%20laptop&limit=10
```

### Product Comparison
```bash
POST /compare/products
{
    "product_ids": ["product_1", "product_2"],
    "comparison_criteria": ["price", "rating", "features"]
}
```

### User Feedback
```bash
POST /feedback
{
    "session_id": "session-uuid",
    "message_id": 123,
    "rating": 4,
    "comment": "Great response!"
}
```

### Analytics
```bash
GET /analytics/weaknesses  # Model improvement insights
GET /analytics/stats       # Usage statistics
```

## Usage Examples

### 1. Product Search
```
User: "Do you have Samsung Galaxy phones in stock?"
Assistant: [Searches products, shows available Samsung Galaxy models with prices and ratings]
```

### 2. Quote Generation
```
User: "Can I get a quote for 5 MacBook Pros and 10 wireless keyboards?"
Assistant: [Generates formatted HTML quote with line items, calculations, and totals]
```

### 3. Product Comparison
```
User: "What's the difference between iPhone 15 and Samsung S24?"
Assistant: [Compares specifications, prices, ratings, and features side-by-side]
```

### 4. Recommendations
```
User: "I need a gaming laptop under $1500"
Assistant: [Searches and ranks gaming laptops, provides recommendations with justifications]
```

## Architecture

```
User Request → FastAPI → Session Manager → LangChain Agent → 
[Vector Search + Reranking + Caching] → LLM Reasoning → Response Formatter → HTML Output
```

### Agent Tools (8 Custom Tools)

1. **`embed_query_params`** - Generate BGE embeddings for queries
2. **`search_vector_store`** - ChromaDB similarity search
3. **`rerank_results`** - BGE reranker for relevance improvement
4. **`compute_cost`** - Price calculations with bulk discounts
5. **`search_item_by_index`** - Direct product lookup by ID
6. **`search_similar_items`** - Semantic similarity search
7. **`create_quote`** - HTML quote generation
8. **`send_html_response`** - Format complete responses

### Caching Strategy

- **Redis**: Active sessions, recent queries (fast access)
- **DiskCache**: Embeddings, search results (persistent)
- **Memory**: Models, frequently accessed data (fastest)

### Session Persistence

SQLite database stores:
- Conversation history
- Message metadata
- Retrieval analytics
- User feedback
- Performance metrics

## Configuration

### Environment Variables
```bash
# Optional configurations
REDIS_HOST=localhost
REDIS_PORT=6379
CHROMA_DB_PATH=./chroma_db
CACHE_DIR=./cache
MAX_CONVERSATION_MEMORY=10
```

### Model Configuration
```python
# In tools/product_rag_tools.py
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # or "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"    # lightweight for PoC
```

## Performance

### Expected Performance (RTX 3090)
- **Embedding Generation**: ~50ms per query (CUDA accelerated)
- **Vector Search**: ~10-20ms for 10K+ products
- **Reranking**: ~30-50ms per query-document pair
- **End-to-end Response**: ~200-500ms for complex queries
- **Throughput**: 100+ concurrent users with caching

### Memory Requirements
- **Models**: ~2GB (BGE-large + reranker + LLM)
- **ChromaDB**: ~500MB (100K products)
- **Cache**: Configurable (default 1GB disk cache)

## Development

### Project Structure
```
Product-RAG/
├── api/                    # FastAPI application
│   └── main.py
├── tools/                  # LangChain tools
│   └── product_rag_tools.py
├── database/               # Session management
│   └── session_manager.py
├── cache/                  # Caching service
│   └── cache_service.py
├── docs/                   # Documentation
│   └── plan.md
├── database_builder.py     # Embedding pipeline
├── startup.py             # Setup & startup script
└── requirements.txt       # Dependencies
```

### Adding New Tools
```python
# In tools/product_rag_tools.py
class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Tool description"
    
    def _run(self, input_param: str) -> str:
        # Tool implementation
        return "result"

# Register in create_product_rag_tools()
```

### Testing
```bash
# Run basic tests
python startup.py --test-only

# Test specific components
python -c "from tools.product_rag_tools import create_product_rag_tools; print('Tools OK')"
python -c "from cache.cache_service import CacheService; CacheService(); print('Cache OK')"
```

## Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y redis-server

# Copy application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Initialize databases
RUN python startup.py --skip-deps --test-only

# Start services
CMD ["python", "startup.py", "--host", "0.0.0.0"]
```

### Production Considerations
- Use PostgreSQL instead of SQLite for high concurrency
- Configure Redis persistence and clustering
- Set up proper logging and monitoring
- Implement rate limiting and authentication
- Use HTTPS with proper SSL certificates

## Analytics & Monitoring

### Model Weakness Analysis
The system automatically tracks:
- Low-performing queries (poor user feedback)
- Frequently ignored products
- Query intent performance
- Response time metrics
- Cache hit/miss ratios

### Dashboard Queries
```sql
-- Top problematic queries
SELECT query, AVG(rating), COUNT(*) 
FROM retrieval_analytics ra
JOIN user_feedback uf ON ra.message_id = uf.message_id
WHERE rating <= 3 GROUP BY query ORDER BY COUNT(*) DESC;

-- Performance bottlenecks
SELECT AVG(embedding_time_ms), AVG(search_time_ms), AVG(rerank_time_ms)
FROM messages WHERE role = 'assistant' AND timestamp > datetime('now', '-7 days');
```

## Troubleshooting

### Common Issues

1. **CUDA not detected**: Install CUDA 11.6+ drivers
2. **Redis connection failed**: Install Redis or use `--skip-redis` flag
3. **Out of memory**: Reduce batch size in `database_builder.py`
4. **Slow responses**: Check cache hit rates, consider upgrading hardware

### Debug Mode
```bash
# Enable verbose logging
python startup.py --log-level DEBUG

# Check component health
curl http://localhost:8000/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:
- Check the `/docs` endpoint for API documentation
- Review logs in `./logs/` directory
- Use the `/analytics/weaknesses` endpoint for performance insights