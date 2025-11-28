"""
Quick database builder - minimal version for testing
"""

import logging
from database_builder import ProductEmbeddingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🚀 Starting database build...")

# Create pipeline
pipeline = ProductEmbeddingPipeline(
    embedding_model="BAAI/bge-large-en-v1.5",
    batch_size=4,  # Very small batch for CPU
    cache_dir="./cache",
    chroma_db_path="./chroma_db"
)

print("✅ Pipeline created")

# Process limited dataset
print("📥 Processing 500 products...")
pipeline.process_dataset(max_products=500)

print("🔍 Testing search...")
results = pipeline.search_products("Dell laptop", k=3)

print("📊 Search results:")
for i, result in enumerate(results, 1):
    title = result['metadata'].get('title', 'No title')[:60]
    score = result['similarity_score']
    print(f"  {i}. {title} (Score: {score:.3f})")

print("✅ Database build complete!")