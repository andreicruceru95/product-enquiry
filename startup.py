"""
Startup script for the Product RAG application.
Handles initialization, dependency installation, and service startup.
"""

import os
import sys
import subprocess
import asyncio
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductRAGStartup:
    """
    Handles complete startup process for the Product RAG application.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.venv_path = self.project_root / ".venv"
        self.python_exe = self._get_python_executable()
        
    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable path."""
        if os.name == 'nt':  # Windows
            if self.venv_path.exists():
                return str(self.venv_path / "Scripts" / "python.exe")
            else:
                return "python"
        else:  # Unix-like
            if self.venv_path.exists():
                return str(self.venv_path / "bin" / "python")
            else:
                return "python3"
    
    def create_virtual_environment(self):
        """Create a virtual environment if it doesn't exist."""
        if self.venv_path.exists():
            logger.info("Virtual environment already exists")
            return
        
        logger.info("Creating virtual environment...")
        try:
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True, cwd=self.project_root)
            logger.info("Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            sys.exit(1)
    
    def install_dependencies(self):
        """Install Python dependencies."""
        logger.info("Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            sys.exit(1)
        
        try:
            # Upgrade pip first
            subprocess.run([
                self.python_exe, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, cwd=self.project_root)
            
            # Install requirements
            subprocess.run([
                self.python_exe, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, cwd=self.project_root)
            
            logger.info("Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            sys.exit(1)
    
    def check_cuda_availability(self):
        """Check if CUDA is available for PyTorch."""
        try:
            result = subprocess.run([
                self.python_exe, "-c", 
                "import torch; print('CUDA Available:', torch.cuda.is_available()); "
                "print('CUDA Devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info(f"CUDA Status: {result.stdout.strip()}")
            else:
                logger.warning(f"Could not check CUDA status: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Error checking CUDA: {e}")
    
    def initialize_database(self):
        """Initialize the ChromaDB and SQLite databases."""
        logger.info("Initializing databases...")
        
        try:
            # Create necessary directories
            (self.project_root / "chroma_db").mkdir(exist_ok=True)
            (self.project_root / "cache").mkdir(exist_ok=True)
            
            # Initialize session database
            init_script = f"""
import asyncio
import sys
import os
sys.path.append({repr(str(self.project_root))})

from database.session_manager import SessionManager

async def init_db():
    sm = SessionManager('conversations.db')
    await sm.init_database()
    print('Session database initialized')

asyncio.run(init_db())
"""
            
            result = subprocess.run([
                self.python_exe, "-c", init_script
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Databases initialized successfully")
            else:
                logger.error(f"Database initialization failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error initializing databases: {e}")
    
    def build_embeddings_database(self, max_products: int = None):
        """Build the embeddings database from Amazon Products dataset."""
        logger.info("Building embeddings database...")
        
        build_script = f"""
import sys
import os
sys.path.append({repr(str(self.project_root))})

from database_builder import ProductEmbeddingPipeline

def main():
    pipeline = ProductEmbeddingPipeline()
    pipeline.process_dataset(max_products={max_products})

if __name__ == "__main__":
    main()
"""
        
        try:
            # Run the embedding pipeline
            process = subprocess.Popen([
                self.python_exe, "-c", build_script
            ], cwd=self.project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               universal_newlines=True)
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    logger.info(f"Embedding Builder: {line.strip()}")
            
            process.wait()
            
            if process.returncode == 0:
                logger.info("Embeddings database built successfully")
            else:
                logger.error("Embeddings database build failed")
                
        except Exception as e:
            logger.error(f"Error building embeddings database: {e}")
    
    def start_redis_server(self):
        """Start Redis server if available."""
        try:
            # Check if Redis is running
            result = subprocess.run([
                "redis-cli", "ping"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Redis server is already running")
                return True
            
            # Try to start Redis
            logger.info("Starting Redis server...")
            if os.name == 'nt':  # Windows
                # On Windows, Redis might be installed via Chocolatey or manual install
                subprocess.Popen(["redis-server"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Unix-like
                subprocess.Popen(["redis-server", "--daemonize", "yes"])
            
            # Wait a moment and check again
            import time
            time.sleep(2)
            result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Redis server started successfully")
                return True
            else:
                logger.warning("Redis server failed to start, using disk cache only")
                return False
                
        except FileNotFoundError:
            logger.warning("Redis not found, using disk cache only")
            return False
        except Exception as e:
            logger.warning(f"Error starting Redis: {e}, using disk cache only")
            return False
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
        """Start the FastAPI server."""
        logger.info(f"Starting API server on {host}:{port}...")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            # Start the server
            cmd = [
                self.python_exe, "-m", "uvicorn", 
                "api.main:app",
                "--host", host,
                "--port", str(port)
            ]
            
            if reload:
                cmd.append("--reload")
            
            subprocess.run(cmd, cwd=self.project_root, env=env)
            
        except KeyboardInterrupt:
            logger.info("API server stopped")
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
    
    def run_tests(self):
        """Run basic tests to verify installation."""
        logger.info("Running basic tests...")
        
        test_script = f"""
import sys
import os
sys.path.append({repr(str(self.project_root))})

# Test imports
try:
    import torch
    import sentence_transformers
    import chromadb
    import langchain
    import fastapi
    from FlagEmbedding import FlagReranker
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {{e}}")
    sys.exit(1)

# Test CUDA
import torch
print(f"✓ CUDA available: {{torch.cuda.is_available()}}")

# Test cache
from cache.cache_service import CacheService
cache = CacheService()
print("✓ Cache service initialized")

# Test tools
from tools.product_rag_tools import create_product_rag_tools
tools = create_product_rag_tools()
print(f"✓ Created {{len(tools)}} RAG tools")

print("All tests passed!")
"""
        
        try:
            result = subprocess.run([
                self.python_exe, "-c", test_script
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("All tests passed!")
                print(result.stdout)
            else:
                logger.error(f"Tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
        
        return True

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="Product RAG Application Startup")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embeddings database build")
    parser.add_argument("--max-products", type=int, help="Limit number of products to process")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--test-only", action="store_true", help="Run tests only")
    
    args = parser.parse_args()
    
    # Initialize startup handler
    startup = ProductRAGStartup()
    
    # Create virtual environment
    startup.create_virtual_environment()
    
    # Install dependencies
    if not args.skip_deps:
        startup.install_dependencies()
    
    # Check CUDA availability
    startup.check_cuda_availability()
    
    # Initialize databases
    startup.initialize_database()
    
    # Build embeddings database
    if not args.skip_embeddings:
        startup.build_embeddings_database(max_products=args.max_products)
    
    # Start Redis server
    startup.start_redis_server()
    
    # Run tests
    if not startup.run_tests():
        logger.error("Tests failed, aborting startup")
        sys.exit(1)
    
    if args.test_only:
        logger.info("Test-only mode, exiting")
        return
    
    # Start API server
    startup.start_api_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )

if __name__ == "__main__":
    main()