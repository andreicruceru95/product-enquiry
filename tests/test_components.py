"""
Test script to verify all components are working before building the full database.
"""

import logging
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_torch_setup():
    """Test PyTorch setup."""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def test_sentence_transformers(device):
    """Test sentence transformers."""
    logger.info("Loading BGE embedding model...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    
    # Test encoding
    test_text = "Dell laptop computer"
    logger.info(f"Testing embedding generation for: '{test_text}'")
    embedding = model.encode([test_text])
    logger.info(f"Generated embedding shape: {embedding.shape}")
    return model

def test_chromadb():
    """Test ChromaDB setup."""
    logger.info("Testing ChromaDB...")
    client = chromadb.PersistentClient(path="./test_chroma_db")
    
    # Create test collection
    collection = client.get_or_create_collection(
        name="test_products",
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info(f"ChromaDB collection created. Current count: {collection.count()}")
    return client, collection

def test_dataset_access():
    """Test access to Amazon Products dataset."""
    logger.info("Testing dataset access...")
    try:
        # Try to load just the first few items
        ds = load_dataset("milistu/AMAZON-Products-2023", split="train[:10]")
        logger.info(f"Successfully loaded {len(ds)} sample products")
        
        # Show first product
        first_product = ds[0]
        logger.info(f"Sample product title: {first_product.get('title', 'No title')}")
        logger.info(f"Sample product keys: {list(first_product.keys())}")
        return True
    except Exception as e:
        logger.error(f"Dataset access failed: {e}")
        return False

def test_basic_pipeline():
    """Test basic embedding pipeline."""
    logger.info("Testing basic embedding pipeline...")
    
    # Test components
    device = test_torch_setup()
    model = test_sentence_transformers(device)
    client, collection = test_chromadb()
    
    # Test dataset access
    if not test_dataset_access():
        logger.error("Dataset access failed - you may need to authenticate with Hugging Face")
        return False
    
    # Add a test document to ChromaDB
    test_doc = "Dell XPS 15 laptop with Intel Core i7 processor"
    test_embedding = model.encode([test_doc])
    
    collection.add(
        embeddings=[test_embedding[0].tolist()],
        documents=[test_doc],
        metadatas=[{"title": "Dell XPS 15", "price": 1299.99}],
        ids=["test_1"]
    )
    
    # Test search
    query = "Dell laptop"
    query_embedding = model.encode([query])
    
    results = collection.query(
        query_embeddings=[query_embedding[0].tolist()],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )
    
    logger.info(f"Search test successful. Found: {results['documents'][0][0]}")
    logger.info(f"Distance: {results['distances'][0][0]}")
    
    return True

if __name__ == "__main__":
    logger.info("Starting component tests...")
    
    try:
        success = test_basic_pipeline()
        if success:
            logger.info("✅ All component tests passed! Ready to build database.")
        else:
            logger.error("❌ Component tests failed.")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()