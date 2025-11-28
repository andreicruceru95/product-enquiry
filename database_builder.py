import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
import hashlib
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional
import diskcache as dc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductEmbeddingPipeline:
    """
    Advanced embedding pipeline for Amazon Products dataset using BGE models with CUDA acceleration.
    Includes caching, batch processing, and ChromaDB integration.
    """
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 batch_size: int = 32,
                 cache_dir: str = "./cache",
                 chroma_db_path: str = "./chroma_db"):
        
        # Initialize device (CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding model with CUDA support
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        logger.info(f"Loaded embedding model: {embedding_model}")
        
        self.batch_size = batch_size
        
        # Initialize disk cache for embeddings
        self.cache = dc.Cache(cache_dir)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="amazon_products_bge",
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model}
        )
        
        logger.info(f"ChromaDB collection ready. Current count: {self.collection.count()}")
    
    def _create_cache_key(self, text: str) -> str:
        """Create deterministic cache key for text."""
        return f"bge_embedding_{hashlib.md5(text.encode()).hexdigest()}"
    
    def _prepare_product_text(self, product: Dict[str, Any]) -> str:
        """
        Create searchable text from product data (title + description + key features).
        """
        parts = []
        
        # Title (required)
        if product.get("title"):
            parts.append(product["title"])
        
        # Description
        if product.get("description"):
            parts.append(product["description"])
        
        # Key features (if available)
        if product.get("features") and isinstance(product["features"], list):
            parts.extend(product["features"][:3])  # Top 3 features
        
        # Categories for context
        if product.get("categories") and isinstance(product["categories"], list):
            parts.extend(product["categories"][:2])  # Top 2 categories
        
        return " | ".join(parts)
    
    def _extract_metadata(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and clean metadata for ChromaDB storage."""
        metadata = {
            "parent_asin": str(product.get("parent_asin", "")),
            "title": str(product.get("title", ""))[:500],  # Limit length
            "price": float(product.get("price", 0.0)) if product.get("price") else 0.0,
            "average_rating": float(product.get("average_rating", 0.0)) if product.get("average_rating") else 0.0,
            "rating_number": int(product.get("rating_number", 0)) if product.get("rating_number") else 0,
            "store": str(product.get("store", ""))[:100],
            "main_category": str(product.get("main_category", ""))[:100],
        }
        
        # Add categories as JSON string (ChromaDB doesn't support lists directly)
        if product.get("categories"):
            metadata["categories"] = json.dumps(product["categories"][:5])  # Top 5 categories
        
        # Add features as JSON string
        if product.get("features"):
            metadata["features"] = json.dumps(product["features"][:5])  # Top 5 features
        
        # Clean None values and convert to strings (ChromaDB requirement)
        cleaned_metadata = {}
        for k, v in metadata.items():
            if v is not None:
                if isinstance(v, (int, float)):
                    cleaned_metadata[k] = v
                else:
                    cleaned_metadata[k] = str(v)
        
        return cleaned_metadata
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts with caching."""
        embeddings = []
        texts_to_embed = []
        cache_keys = []
        
        # Check cache first
        for text in texts:
            cache_key = self._create_cache_key(text)
            cached_embedding = self.cache.get(cache_key)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                cache_keys.append(None)  # Mark as cached
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                cache_keys.append(cache_key)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} texts...")
            new_embeddings = self.embedding_model.encode(
                texts_to_embed, 
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Cache new embeddings and fill placeholders
            new_idx = 0
            for i, (embedding, cache_key) in enumerate(zip(embeddings, cache_keys)):
                if embedding is None:  # This was a placeholder
                    new_embedding = new_embeddings[new_idx]
                    embeddings[i] = new_embedding
                    
                    # Cache the embedding
                    if cache_key:
                        self.cache.set(cache_key, new_embedding, expire=None)
                    
                    new_idx += 1
        
        return np.array(embeddings)
    
    def process_dataset(self, dataset_name: str = "milistu/AMAZON-Products-2023", 
                       max_products: Optional[int] = None):
        """
        Process the Amazon Products dataset and populate ChromaDB.
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset(dataset_name)
        
        # Get the train split (main data)
        products = ds["train"]
        
        if max_products:
            products = products.select(range(min(max_products, len(products))))
        
        logger.info(f"Processing {len(products)} products...")
        
        # Process in batches
        total_batches = (len(products) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(products))
            
            batch_products = products.select(range(start_idx, end_idx))
            
            # Prepare texts and metadata
            texts = []
            metadatas = []
            ids = []
            
            for i, product in enumerate(batch_products):
                # Create searchable text
                product_text = self._prepare_product_text(product)
                
                # Skip products with no meaningful text
                if not product_text.strip():
                    continue
                
                texts.append(product_text)
                metadatas.append(self._extract_metadata(product))
                ids.append(f"product_{start_idx + i}")
            
            if not texts:
                continue
            
            # Generate embeddings for batch
            embeddings = self.generate_embeddings_batch(texts)
            
            # Add to ChromaDB
            try:
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added batch {batch_idx + 1}/{total_batches} to ChromaDB")
            
            except Exception as e:
                logger.error(f"Error adding batch {batch_idx + 1} to ChromaDB: {e}")
                continue
        
        logger.info(f"Processing complete! Total products in ChromaDB: {self.collection.count()}")
    
    def search_products(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search products using BGE embeddings."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return search_results

def main():
    """Main function to run the embedding pipeline."""
    # Initialize pipeline
    pipeline = ProductEmbeddingPipeline(
        embedding_model="BAAI/bge-large-en-v1.5",
        batch_size=32 if torch.cuda.is_available() else 16
    )
    
    # Process the dataset
    pipeline.process_dataset(max_products=None)  # Process all products
    
    # Test search
    test_queries = [
        "Dell laptop computer",
        "Samsung smartphone Galaxy",
        "wireless headphones bluetooth"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")
        results = pipeline.search_products(query, k=5)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['metadata'].get('title', 'No title')} "
                  f"(Score: {result['similarity_score']:.3f})")

if __name__ == "__main__":
    main()