import aiosqlite
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages conversation sessions, message history, and retrieval analytics
    for model weakness analysis and conversation persistence.
    """
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
    
    async def init_database(self):
        """Initialize the database with required tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
                -- Conversation sessions
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'closed', 'archived'))
                );
                
                -- Individual messages in conversations
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    
                    -- Analysis fields for weakness detection
                    retrieval_documents JSON,
                    reranker_scores JSON,
                    user_feedback INTEGER CHECK (user_feedback BETWEEN 1 AND 5),
                    response_time_ms INTEGER,
                    token_count INTEGER,
                    
                    -- Performance metrics
                    embedding_time_ms INTEGER,
                    search_time_ms INTEGER,
                    rerank_time_ms INTEGER,
                    llm_time_ms INTEGER
                );
                
                -- Product-specific retrieval analytics
                CREATE TABLE IF NOT EXISTS retrieval_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
                    query TEXT NOT NULL,
                    query_intent TEXT,  -- categorized intent (quote, search, comparison, etc.)
                    
                    -- Retrieval results
                    retrieved_product_ids JSON,
                    embedding_scores JSON,
                    reranker_scores JSON,
                    final_ranking JSON,
                    
                    -- User interaction tracking
                    user_interaction TEXT CHECK (user_interaction IN ('clicked', 'purchased', 'ignored', 'modified')),
                    interaction_products JSON,  -- which products user interacted with
                    
                    -- Performance metrics
                    total_products_found INTEGER,
                    search_precision REAL,
                    search_recall REAL,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- User feedback and ratings
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
                    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('rating', 'correction', 'suggestion')),
                    feedback_value TEXT NOT NULL,
                    feedback_metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- System performance metrics
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    metadata JSON,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for better performance
                CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_activity ON conversations(last_activity);
                
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
                CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
                
                CREATE INDEX IF NOT EXISTS idx_retrieval_query ON retrieval_analytics(query);
                CREATE INDEX IF NOT EXISTS idx_retrieval_intent ON retrieval_analytics(query_intent);
                CREATE INDEX IF NOT EXISTS idx_retrieval_interaction ON retrieval_analytics(user_interaction);
                CREATE INDEX IF NOT EXISTS idx_retrieval_created_at ON retrieval_analytics(created_at);
                
                CREATE INDEX IF NOT EXISTS idx_feedback_session ON user_feedback(session_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_type ON user_feedback(feedback_type);
                
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_metrics_recorded_at ON system_metrics(recorded_at);
            """)
            await db.commit()
            logger.info("Database initialized successfully")
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())
    
    async def create_conversation(self, 
                                session_id: Optional[str] = None,
                                user_id: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new conversation session."""
        if session_id is None:
            session_id = self.generate_session_id()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversations (session_id, user_id, metadata)
                VALUES (?, ?, ?)
            """, (session_id, user_id, json.dumps(metadata or {})))
            await db.commit()
        
        logger.info(f"Created conversation session: {session_id}")
        return session_id
    
    async def add_message(self,
                         session_id: str,
                         role: str,
                         content: str,
                         metadata: Optional[Dict[str, Any]] = None,
                         retrieval_data: Optional[Dict[str, Any]] = None,
                         performance_metrics: Optional[Dict[str, int]] = None) -> int:
        """Add a message to the conversation."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get conversation ID
            async with db.execute("""
                SELECT id FROM conversations WHERE session_id = ?
            """, (session_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise ValueError(f"Session {session_id} not found")
                conversation_id = row[0]
            
            # Insert message
            cursor = await db.execute("""
                INSERT INTO messages (
                    conversation_id, role, content, metadata,
                    retrieval_documents, reranker_scores,
                    embedding_time_ms, search_time_ms, rerank_time_ms, llm_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, role, content, json.dumps(metadata or {}),
                json.dumps(retrieval_data.get('documents', []) if retrieval_data else []),
                json.dumps(retrieval_data.get('reranker_scores', []) if retrieval_data else []),
                performance_metrics.get('embedding_time_ms', 0) if performance_metrics else 0,
                performance_metrics.get('search_time_ms', 0) if performance_metrics else 0,
                performance_metrics.get('rerank_time_ms', 0) if performance_metrics else 0,
                performance_metrics.get('llm_time_ms', 0) if performance_metrics else 0
            ))
            
            message_id = cursor.lastrowid
            
            # Update conversation last_activity
            await db.execute("""
                UPDATE conversations 
                SET last_activity = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            """, (session_id,))
            
            await db.commit()
        
        return message_id
    
    async def add_retrieval_analytics(self,
                                    message_id: int,
                                    query: str,
                                    query_intent: str,
                                    retrieval_results: Dict[str, Any],
                                    performance_data: Optional[Dict[str, Any]] = None):
        """Add retrieval analytics for model improvement."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO retrieval_analytics (
                    message_id, query, query_intent,
                    retrieved_product_ids, embedding_scores, reranker_scores, final_ranking,
                    total_products_found
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id, query, query_intent,
                json.dumps(retrieval_results.get('product_ids', [])),
                json.dumps(retrieval_results.get('embedding_scores', [])),
                json.dumps(retrieval_results.get('reranker_scores', [])),
                json.dumps(retrieval_results.get('final_ranking', [])),
                retrieval_results.get('total_found', 0)
            ))
            await db.commit()
    
    async def get_conversation_history(self, 
                                     session_id: str, 
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT m.role, m.content, m.timestamp, m.metadata
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.session_id = ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            """, (session_id, limit)) as cursor:
                rows = await cursor.fetchall()
        
        messages = []
        for row in rows:
            messages.append({
                'role': row[0],
                'content': row[1],
                'timestamp': row[2],
                'metadata': json.loads(row[3]) if row[3] else {}
            })
        
        return list(reversed(messages))  # Return in chronological order
    
    async def add_user_feedback(self,
                              session_id: str,
                              message_id: int,
                              feedback_type: str,
                              feedback_value: str,
                              metadata: Optional[Dict[str, Any]] = None):
        """Add user feedback for model improvement."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO user_feedback (session_id, message_id, feedback_type, feedback_value, feedback_metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, message_id, feedback_type, feedback_value, json.dumps(metadata or {})))
            await db.commit()
    
    async def get_model_weaknesses(self) -> Dict[str, Any]:
        """Analyze conversation data to identify model weaknesses."""
        async with aiosqlite.connect(self.db_path) as db:
            analysis = {}
            
            # Low-performing queries (low user feedback)
            async with db.execute("""
                SELECT ra.query, AVG(CAST(uf.feedback_value AS REAL)) as avg_rating, COUNT(*) as frequency
                FROM retrieval_analytics ra
                JOIN user_feedback uf ON ra.message_id = uf.message_id
                WHERE uf.feedback_type = 'rating' AND CAST(uf.feedback_value AS REAL) <= 3
                GROUP BY ra.query
                HAVING COUNT(*) >= 3
                ORDER BY avg_rating ASC, frequency DESC
                LIMIT 20
            """) as cursor:
                analysis['low_performing_queries'] = await cursor.fetchall()
            
            # Frequently ignored products
            async with db.execute("""
                SELECT ra.retrieved_product_ids, COUNT(*) as ignore_frequency
                FROM retrieval_analytics ra
                WHERE ra.user_interaction = 'ignored'
                GROUP BY ra.retrieved_product_ids
                ORDER BY ignore_frequency DESC
                LIMIT 20
            """) as cursor:
                analysis['ignored_products'] = await cursor.fetchall()
            
            # Query intents with poor performance
            async with db.execute("""
                SELECT ra.query_intent, AVG(CAST(uf.feedback_value AS REAL)) as avg_rating, COUNT(*) as frequency
                FROM retrieval_analytics ra
                JOIN user_feedback uf ON ra.message_id = uf.message_id
                WHERE uf.feedback_type = 'rating'
                GROUP BY ra.query_intent
                HAVING COUNT(*) >= 5
                ORDER BY avg_rating ASC
            """) as cursor:
                analysis['poor_intent_performance'] = await cursor.fetchall()
            
            # Response time statistics
            async with db.execute("""
                SELECT 
                    AVG(embedding_time_ms) as avg_embedding_time,
                    AVG(search_time_ms) as avg_search_time,
                    AVG(rerank_time_ms) as avg_rerank_time,
                    AVG(llm_time_ms) as avg_llm_time,
                    MAX(embedding_time_ms + search_time_ms + rerank_time_ms + llm_time_ms) as max_total_time
                FROM messages
                WHERE role = 'assistant' AND timestamp > datetime('now', '-7 days')
            """) as cursor:
                analysis['performance_stats'] = await cursor.fetchone()
            
            return analysis
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a specific conversation and all its messages."""
        async with aiosqlite.connect(self.db_path) as db:
            # Check if session exists
            async with db.execute("""
                SELECT id FROM conversations WHERE session_id = ?
            """, (session_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return False
                
                conversation_id = row[0]
            
            # Delete messages (will cascade to retrieval_analytics and user_feedback)
            await db.execute("""
                DELETE FROM messages WHERE conversation_id = ?
            """, (conversation_id,))
            
            # Delete conversation
            await db.execute("""
                DELETE FROM conversations WHERE session_id = ?
            """, (session_id,))
            
            await db.commit()
            logger.info(f"Deleted conversation session: {session_id}")
            return True

    async def get_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation details by session ID."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT session_id, user_id, started_at, last_activity, metadata, status
                FROM conversations 
                WHERE session_id = ?
            """, (session_id,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return {
                    'session_id': row[0],
                    'user_id': row[1],
                    'started_at': row[2],
                    'last_activity': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {},
                    'status': row[5]
                }

    async def get_conversation_messages(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get all messages for a conversation."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT m.role, m.content, m.timestamp, m.metadata
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.session_id = ?
                ORDER BY m.timestamp ASC
            """, (session_id,)) as cursor:
                rows = await cursor.fetchall()
                
                if not rows:
                    return None
                
                messages = []
                for row in rows:
                    messages.append({
                        'role': row[0],
                        'content': row[1],
                        'timestamp': row[2],
                        'metadata': json.loads(row[3]) if row[3] else {}
                    })
                
                return messages

    async def cleanup_old_sessions(self, days: int = 30):
        """Clean up old conversation sessions."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE conversations 
                SET status = 'archived'
                WHERE last_activity < datetime('now', '-{} days') AND status = 'active'
            """.format(days))
            await db.commit()
            
            logger.info(f"Archived conversations older than {days} days")
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}
            
            # Total conversations
            async with db.execute("SELECT COUNT(*) FROM conversations") as cursor:
                stats['total_conversations'] = (await cursor.fetchone())[0]
            
            # Active conversations (last 24 hours)
            async with db.execute("""
                SELECT COUNT(*) FROM conversations 
                WHERE last_activity > datetime('now', '-1 day')
            """) as cursor:
                stats['active_conversations_24h'] = (await cursor.fetchone())[0]
            
            # Total messages
            async with db.execute("SELECT COUNT(*) FROM messages") as cursor:
                stats['total_messages'] = (await cursor.fetchone())[0]
            
            # Average conversation length
            async with db.execute("""
                SELECT AVG(message_count) FROM (
                    SELECT COUNT(*) as message_count 
                    FROM messages 
                    GROUP BY conversation_id
                )
            """) as cursor:
                stats['avg_conversation_length'] = (await cursor.fetchone())[0]
            
            return stats

# Usage example and testing functions
async def test_session_manager():
    """Test the session manager functionality."""
    sm = SessionManager("test_conversations.db")
    await sm.init_database()
    
    # Create a test conversation
    session_id = await sm.create_conversation(user_id="test_user")
    
    # Add messages
    msg_id = await sm.add_message(session_id, "user", "I need a Dell laptop")
    await sm.add_message(session_id, "assistant", "Here are some Dell laptop options...")
    
    # Add retrieval analytics
    await sm.add_retrieval_analytics(
        message_id=msg_id,
        query="Dell laptop",
        query_intent="product_search",
        retrieval_results={
            'product_ids': ['prod_1', 'prod_2'],
            'embedding_scores': [0.9, 0.8],
            'final_ranking': ['prod_1', 'prod_2']
        }
    )
    
    # Add user feedback
    await sm.add_user_feedback(session_id, msg_id, "rating", "4")
    
    # Get conversation history
    history = await sm.get_conversation_history(session_id)
    print(f"Conversation history: {len(history)} messages")
    
    # Get stats
    stats = await sm.get_session_stats()
    print(f"Session stats: {stats}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_session_manager())