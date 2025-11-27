"""
LangChain tools for the Product RAG application.
Implements 8 specialized tools for product search, reranking, cost calculation, and quote generation.
"""

import json
import time
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

from langchain.tools import BaseTool
from langchain.schema import Document
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import chromadb
import diskcache as dc
from jinja2 import Template

logger = logging.getLogger(__name__)

# Pydantic models for tool inputs
class EmbedQueryInput(BaseModel):
    q: str = Field(description="Query string to embed")

class SearchVectorStoreInput(BaseModel):
    embeddings: List[float] = Field(description="Query embeddings as list of floats")
    k: int = Field(default=20, description="Number of results to return")

class RerankResultsInput(BaseModel):
    res: List[str] = Field(description="List of result documents to rerank")
    q: str = Field(description="Original query for reranking")

class ComputeCostInput(BaseModel):
    price: float = Field(description="Unit price of the product")
    item_numbers: int = Field(description="Number of items")

class SearchItemByIndexInput(BaseModel):
    idx: str = Field(description="Product index/ID to search for")

class SearchSimilarItemsInput(BaseModel):
    query: str = Field(description="Query to find similar products")

class CreateQuoteInput(BaseModel):
    products: Dict[str, Any] = Field(description="Dictionary of products with quantities and details")

class SendHtmlResponseInput(BaseModel):
    response: str = Field(description="HTML response to format and send")

class ProductRAGToolkit:
    """
    Toolkit containing all 8 tools for the Product RAG application.
    Handles caching, embeddings, search, reranking, and quote generation.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-large-en-v1.5",
                 reranker_model_name: str = "BAAI/bge-reranker-base",
                 chroma_db_path: str = "./chroma_db",
                 cache_dir: str = "./cache"):
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker = FlagReranker(reranker_model_name, use_fp16=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        try:
            self.collection = self.chroma_client.get_collection("amazon_products_bge")
        except Exception as e:
            logger.error(f"ChromaDB collection not found: {e}")
            self.collection = None
        
        # Initialize cache
        self.cache = dc.Cache(cache_dir)
        
        # HTML templates for responses
        self._setup_templates()
        
        logger.info("ProductRAGToolkit initialized successfully")
    
    def _setup_templates(self):
        """Setup Jinja2 templates for HTML responses."""
        
        # Product card template
        self.product_card_template = Template("""
        <div class="product-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px;">
            <h3 style="color: #333; margin: 0 0 10px 0;">{{ title }}</h3>
            <div class="product-info" style="display: flex; justify-content: space-between; align-items: center;">
                <div class="price" style="font-size: 1.2em; font-weight: bold; color: #e47911;">
                    ${{ "%.2f"|format(price) }}
                </div>
                {% if average_rating %}
                <div class="rating" style="color: #ff9900;">
                    ⭐ {{ average_rating }}/5 ({{ rating_number }} reviews)
                </div>
                {% endif %}
            </div>
            {% if description %}
            <p style="color: #666; margin: 10px 0;">{{ description[:200] }}{% if description|length > 200 %}...{% endif %}</p>
            {% endif %}
            {% if store %}
            <div class="store" style="color: #007185; font-size: 0.9em;">Sold by: {{ store }}</div>
            {% endif %}
        </div>
        """)
        
        # Quote table template
        self.quote_template = Template("""
        <div class="quote-container" style="max-width: 800px; margin: 20px auto; font-family: Arial, sans-serif;">
            <h2 style="color: #232f3e; text-align: center; margin-bottom: 20px;">Product Quote</h2>
            
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Product</th>
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Quantity</th>
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: right;">Unit Price</th>
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: right;">Total</th>
                    </tr>
                </thead>
                <tbody>
                    {% for item in products %}
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 12px;">
                            <div style="font-weight: bold;">{{ item.title }}</div>
                            {% if item.store %}
                            <div style="color: #666; font-size: 0.9em;">by {{ item.store }}</div>
                            {% endif %}
                        </td>
                        <td style="border: 1px solid #ddd; padding: 12px; text-align: center;">{{ item.quantity }}</td>
                        <td style="border: 1px solid #ddd; padding: 12px; text-align: right;">${{ "%.2f"|format(item.unit_price) }}</td>
                        <td style="border: 1px solid #ddd; padding: 12px; text-align: right; font-weight: bold;">${{ "%.2f"|format(item.total_price) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
                <tfoot>
                    <tr style="background-color: #f8f9fa; font-weight: bold;">
                        <td colspan="3" style="border: 1px solid #ddd; padding: 12px; text-align: right;">Grand Total:</td>
                        <td style="border: 1px solid #ddd; padding: 12px; text-align: right; color: #e47911; font-size: 1.1em;">
                            ${{ "%.2f"|format(total_amount) }}
                        </td>
                    </tr>
                </tfoot>
            </table>
            
            <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 20px;">
                Generated on {{ timestamp }}
            </div>
        </div>
        """)
        
        # Full response template
        self.response_template = Template("""
        <div class="chat-response" style="max-width: 900px; margin: 0 auto; font-family: Arial, sans-serif;">
            {{ greeting }}
            {{ content }}
            {{ end_message }}
        </div>
        """)

class EmbedQueryTool(BaseTool):
    """Tool to generate embeddings for query parameters."""
    
    name = "embed_query_params"
    description = "Generate embeddings for a query string using BGE model"
    args_schema = EmbedQueryInput
    
    def __init__(self, toolkit: ProductRAGToolkit):
        super().__init__()
        self.toolkit = toolkit
    
    def _run(self, q: str) -> List[float]:
        """Generate embeddings for the query."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"embed_bge_{hashlib.md5(q.encode()).hexdigest()}"
        cached_embedding = self.toolkit.cache.get(cache_key)
        
        if cached_embedding is not None:
            logger.info(f"Cache hit for query embedding: {q[:50]}...")
            return cached_embedding.tolist()
        
        # Generate embedding
        embedding = self.toolkit.embedding_model.encode([q], normalize_embeddings=True)[0]
        
        # Cache the result
        self.toolkit.cache.set(cache_key, embedding, expire=3600)  # 1 hour TTL
        
        end_time = time.time()
        logger.info(f"Generated embedding for query: {q[:50]}... ({end_time - start_time:.3f}s)")
        
        return embedding.tolist()

class SearchVectorStoreTool(BaseTool):
    """Tool to search the ChromaDB vector store."""
    
    name = "search_vector_store"
    description = "Search the product vector store using embeddings"
    args_schema = SearchVectorStoreInput
    
    def __init__(self, toolkit: ProductRAGToolkit):
        super().__init__()
        self.toolkit = toolkit
    
    def _run(self, embeddings: List[float], k: int = 20) -> List[Dict[str, Any]]:
        """Search the vector store with the provided embeddings."""
        if not self.toolkit.collection:
            return {"error": "ChromaDB collection not available"}
        
        start_time = time.time()
        
        try:
            # Search ChromaDB
            results = self.toolkit.collection.query(
                query_embeddings=[embeddings],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],
                    "distance": results["distances"][0][i]
                }
                search_results.append(result)
            
            end_time = time.time()
            logger.info(f"Vector search completed: {len(search_results)} results ({end_time - start_time:.3f}s)")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"error": str(e)}

class RerankResultsTool(BaseTool):
    """Tool to rerank search results using BGE reranker."""
    
    name = "rerank_results"
    description = "Rerank search results using BGE reranker for improved relevance"
    args_schema = RerankResultsInput
    
    def __init__(self, toolkit: ProductRAGToolkit):
        super().__init__()
        self.toolkit = toolkit
    
    def _run(self, res: List[str], q: str) -> List[Dict[str, Any]]:
        """Rerank results based on query relevance."""
        start_time = time.time()
        
        if not res:
            return []
        
        try:
            # Prepare query-document pairs for reranker
            pairs = [[q, doc] for doc in res]
            
            # Get reranker scores
            scores = self.toolkit.reranker.compute_score(pairs, normalize=True)
            
            # Create scored results
            reranked_results = []
            for i, (doc, score) in enumerate(zip(res, scores)):
                reranked_results.append({
                    "original_index": i,
                    "document": doc,
                    "rerank_score": float(score),
                    "rerank_rank": 0  # Will be set after sorting
                })
            
            # Sort by reranker score
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result["rerank_rank"] = i + 1
            
            end_time = time.time()
            logger.info(f"Reranked {len(res)} results ({end_time - start_time:.3f}s)")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return [{"document": doc, "error": str(e)} for doc in res]

class ComputeCostTool(BaseTool):
    """Tool to compute total cost for products."""
    
    name = "compute_cost"
    description = "Calculate total cost based on price and quantity"
    args_schema = ComputeCostInput
    
    def _run(self, price: float, item_numbers: int) -> Dict[str, Any]:
        """Compute cost with potential bulk discounts."""
        try:
            unit_price = float(price)
            quantity = int(item_numbers)
            
            # Base calculation
            subtotal = unit_price * quantity
            
            # Apply bulk discounts
            discount_rate = 0.0
            if quantity >= 100:
                discount_rate = 0.15  # 15% discount for 100+ items
            elif quantity >= 50:
                discount_rate = 0.10  # 10% discount for 50+ items
            elif quantity >= 20:
                discount_rate = 0.05  # 5% discount for 20+ items
            
            discount_amount = subtotal * discount_rate
            total_cost = subtotal - discount_amount
            
            # Tax calculation (example: 8.5% sales tax)
            tax_rate = 0.085
            tax_amount = total_cost * tax_rate
            final_total = total_cost + tax_amount
            
            result = {
                "unit_price": unit_price,
                "quantity": quantity,
                "subtotal": subtotal,
                "discount_rate": discount_rate,
                "discount_amount": discount_amount,
                "discounted_total": total_cost,
                "tax_rate": tax_rate,
                "tax_amount": tax_amount,
                "final_total": final_total,
                "savings": discount_amount
            }
            
            logger.info(f"Cost computed: {quantity} items @ ${unit_price} = ${final_total:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Cost computation error: {e}")
            return {"error": str(e)}

class SearchItemByIndexTool(BaseTool):
    """Tool to search for a specific product by index/ID."""
    
    name = "search_item_by_index"
    description = "Find a specific product by its index or ID"
    args_schema = SearchItemByIndexInput
    
    def __init__(self, toolkit: ProductRAGToolkit):
        super().__init__()
        self.toolkit = toolkit
    
    def _run(self, idx: str) -> Dict[str, Any]:
        """Search for a product by its index/ID."""
        if not self.toolkit.collection:
            return {"error": "ChromaDB collection not available"}
        
        try:
            # Search by ID
            result = self.toolkit.collection.get(
                ids=[idx],
                include=["documents", "metadatas"]
            )
            
            if result["ids"]:
                product_data = {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0],
                    "found": True
                }
                logger.info(f"Product found by index: {idx}")
                return product_data
            else:
                logger.info(f"Product not found by index: {idx}")
                return {"id": idx, "found": False, "error": "Product not found"}
                
        except Exception as e:
            logger.error(f"Index search error: {e}")
            return {"id": idx, "error": str(e)}

class SearchSimilarItemsTool(BaseTool):
    """Tool to find similar products."""
    
    name = "search_similar_items"
    description = "Find products similar to a given query or product"
    args_schema = SearchSimilarItemsInput
    
    def __init__(self, toolkit: ProductRAGToolkit):
        super().__init__()
        self.toolkit = toolkit
    
    def _run(self, query: str) -> List[Dict[str, Any]]:
        """Find similar products using semantic search."""
        if not self.toolkit.collection:
            return [{"error": "ChromaDB collection not available"}]
        
        try:
            # Generate embedding for similarity search
            embedding = self.toolkit.embedding_model.encode([query], normalize_embeddings=True)[0]
            
            # Search for similar items
            results = self.toolkit.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=10,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format similar items
            similar_items = []
            for i in range(len(results["ids"][0])):
                item = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i]
                }
                similar_items.append(item)
            
            logger.info(f"Found {len(similar_items)} similar items for query: {query[:50]}...")
            return similar_items
            
        except Exception as e:
            logger.error(f"Similar items search error: {e}")
            return [{"error": str(e)}]

class CreateQuoteTool(BaseTool):
    """Tool to create HTML quotes for products."""
    
    name = "create_quote"
    description = "Generate an HTML quote table for selected products"
    args_schema = CreateQuoteInput
    
    def __init__(self, toolkit: ProductRAGToolkit):
        super().__init__()
        self.toolkit = toolkit
    
    def _run(self, products: Dict[str, Any]) -> str:
        """Create an HTML quote from product data."""
        try:
            # Process products data
            quote_items = []
            total_amount = 0.0
            
            for product_data in products.get('items', []):
                quantity = product_data.get('quantity', 1)
                unit_price = float(product_data.get('price', 0))
                total_price = unit_price * quantity
                
                quote_items.append({
                    'title': product_data.get('title', 'Unknown Product'),
                    'store': product_data.get('store', ''),
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'total_price': total_price
                })
                
                total_amount += total_price
            
            # Generate quote HTML
            quote_html = self.toolkit.quote_template.render(
                products=quote_items,
                total_amount=total_amount,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            logger.info(f"Generated quote for {len(quote_items)} products, total: ${total_amount:.2f}")
            return quote_html
            
        except Exception as e:
            logger.error(f"Quote generation error: {e}")
            return f"<div style='color: red;'>Error generating quote: {e}</div>"

class SendHtmlResponseTool(BaseTool):
    """Tool to format and send HTML responses."""
    
    name = "send_html_response"
    description = "Format a complete HTML response with greeting and end message"
    args_schema = SendHtmlResponseInput
    
    def __init__(self, toolkit: ProductRAGToolkit):
        super().__init__()
        self.toolkit = toolkit
    
    def _run(self, response: str) -> str:
        """Format a complete HTML response."""
        try:
            # Standard greeting and closing messages
            greeting = """
            <div style="margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                <h3 style="color: #232f3e; margin: 0;">Thank you for your inquiry!</h3>
                <p style="margin: 5px 0 0 0; color: #666;">Here's the information you requested:</p>
            </div>
            """
            
            end_message = """
            <div style="margin-top: 20px; padding: 15px; background-color: #e8f5e8; border-radius: 8px;">
                <p style="margin: 0; color: #2d5a2d;">
                    <strong>Need help with your order?</strong><br>
                    Contact our sales team for personalized assistance or to place your order.
                    We're here to help you find exactly what you need!
                </p>
            </div>
            """
            
            # Combine all parts
            full_response = self.toolkit.response_template.render(
                greeting=greeting,
                content=response,
                end_message=end_message
            )
            
            logger.info("Formatted complete HTML response")
            return full_response
            
        except Exception as e:
            logger.error(f"Response formatting error: {e}")
            return f"<div style='color: red;'>Error formatting response: {e}</div>"

def create_product_rag_tools(embedding_model: str = "BAAI/bge-large-en-v1.5",
                           reranker_model: str = "BAAI/bge-reranker-base",
                           chroma_db_path: str = "./chroma_db",
                           cache_dir: str = "./cache") -> List[BaseTool]:
    """
    Create all 8 Product RAG tools with shared toolkit.
    
    Returns:
        List of configured LangChain tools
    """
    # Initialize shared toolkit
    toolkit = ProductRAGToolkit(
        embedding_model_name=embedding_model,
        reranker_model_name=reranker_model,
        chroma_db_path=chroma_db_path,
        cache_dir=cache_dir
    )
    
    # Create all tools
    tools = [
        EmbedQueryTool(toolkit),
        SearchVectorStoreTool(toolkit),
        RerankResultsTool(toolkit),
        ComputeCostTool(),  # Doesn't need toolkit
        SearchItemByIndexTool(toolkit),
        SearchSimilarItemsTool(toolkit),
        CreateQuoteTool(toolkit),
        SendHtmlResponseTool(toolkit)
    ]
    
    logger.info(f"Created {len(tools)} Product RAG tools")
    return tools

# Example usage and testing
if __name__ == "__main__":
    # Create tools
    tools = create_product_rag_tools()
    
    # Test embed tool
    embed_tool = tools[0]
    embeddings = embed_tool._run("Dell laptop computer")
    print(f"Generated embeddings length: {len(embeddings)}")
    
    # Test search tool
    search_tool = tools[1]
    search_results = search_tool._run(embeddings, k=5)
    print(f"Search results: {len(search_results)} items")
    
    # Test cost computation
    cost_tool = tools[3]
    cost_result = cost_tool._run(899.99, 10)
    print(f"Cost calculation: ${cost_result.get('final_total', 0):.2f}")