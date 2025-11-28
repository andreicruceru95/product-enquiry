"""
Database builder script with progress monitoring and error handling.
"""

import logging
import time
from database_builder import ProductEmbeddingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_database(max_products=5000):
    """Build the embeddings database with progress monitoring."""
    
    logger.info(f"Starting database build for {max_products} products...")
    start_time = time.time()
    
    try:
        # Initialize pipeline with CPU-optimized settings
        pipeline = ProductEmbeddingPipeline(
            embedding_model="BAAI/bge-large-en-v1.5",
            batch_size=8,  # Smaller batch for CPU
            cache_dir="./cache",
            chroma_db_path="./chroma_db"
        )
        
        logger.info("Pipeline initialized successfully")
        
        # Process dataset
        pipeline.process_dataset(max_products=max_products)
        
        # Test search functionality
        logger.info("Testing search functionality...")
        test_queries = [
            "Dell laptop computer",
            "Samsung smartphone",
            "wireless headphones bluetooth",
            "gaming keyboard mechanical",
            "USB cable charger"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            results = pipeline.search_products(query, k=3)
            
            for i, result in enumerate(results, 1):
                title = result['metadata'].get('title', 'No title')[:50]
                score = result['similarity_score']
                logger.info(f"  {i}. {title}... (Score: {score:.3f})")
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✅ Database build completed successfully!")
        logger.info(f"⏱️  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"📊 Products per second: {max_products/duration:.2f}")
        logger.info(f"💾 Database location: ./chroma_db")
        logger.info(f"🗄️  Cache location: ./cache")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Product RAG Database")
    parser.add_argument("--max-products", type=int, default=5000, 
                       help="Maximum number of products to process (default: 5000)")
    
    args = parser.parse_args()
    
    success = build_database(max_products=args.max_products)
    
    if success:
        print("\n🎉 Ready to start the API server!")
        print("Run: python startup.py --skip-embeddings")
        print("Or: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("\n❌ Database build failed. Check logs above.")