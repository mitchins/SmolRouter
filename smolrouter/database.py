import os
import logging
import re
from datetime import datetime, timedelta
from peewee import (
    AutoField,
    CharField,
    DateTimeField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)
from smolrouter.storage import get_blob_storage

logger = logging.getLogger("model-rerouter")

# Database configuration
DB_PATH = os.getenv("DB_PATH", "requests.db")
MAX_AGE_DAYS = int(os.getenv("MAX_LOG_AGE_DAYS", "7"))

# Initialize database
db = SqliteDatabase(DB_PATH)

def estimate_token_count(text: str) -> int:
    """Estimate token count for text content.
    
    Uses a simple but reasonably accurate heuristic:
    - Split on whitespace and punctuation
    - Count tokens roughly as words + punctuation marks
    - Better than simple character/4 for real-world text
    
    Args:
        text: Input text to count tokens for
        
    Returns:
        Estimated token count
    """
    if not text or not isinstance(text, str):
        return 0
    
    # Split on whitespace and common punctuation
    # This gives a decent approximation for most text
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return len(tokens)

def extract_tokens_from_openai_response(response_data: dict) -> tuple[int, int, int]:
    """Extract token counts from OpenAI response usage data.
    
    Args:
        response_data: OpenAI API response dictionary
        
    Returns:
        Tuple of (prompt_tokens, completion_tokens, total_tokens)
        Returns (0, 0, 0) if usage data not available
    """
    usage = response_data.get('usage', {})
    if not usage:
        return (0, 0, 0)
    
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0) 
    total_tokens = usage.get('total_tokens', 0)
    
    return (prompt_tokens, completion_tokens, total_tokens)

def estimate_tokens_from_request(request_data: dict) -> int:
    """Estimate token count from request payload.
    
    Args:
        request_data: Request payload dictionary
        
    Returns:
        Estimated prompt token count
    """
    if not request_data:
        return 0
        
    total_text = ""
    
    # Handle OpenAI format
    if 'messages' in request_data:
        for message in request_data.get('messages', []):
            if isinstance(message, dict) and 'content' in message:
                total_text += str(message['content']) + " "
    
    # Handle legacy prompt format
    if 'prompt' in request_data:
        total_text += str(request_data['prompt']) + " "
    
    return estimate_token_count(total_text)

class BaseModel(Model):
    class Meta:
        database = db

class RequestLog(BaseModel):
    """Log of all requests and responses through the proxy"""
    id = AutoField()
    timestamp = DateTimeField(default=datetime.now)
    
    # Request details
    source_ip = CharField(max_length=45)  # IPv6 compatible
    method = CharField(max_length=10)
    path = CharField(max_length=500)
    service_type = CharField(max_length=10)  # 'openai' or 'ollama'
    
    # Upstream details
    upstream_url = CharField(max_length=500)
    
    # Enhanced traceability
    request_id = CharField(max_length=64, null=True)  # Unique request identifier
    user_agent = CharField(max_length=500, null=True)
    auth_user = CharField(max_length=100, null=True)  # From JWT payload
    
    # Model mapping
    original_model = CharField(max_length=200, null=True)
    mapped_model = CharField(max_length=200, null=True)
    
    # Performance metrics
    duration_ms = IntegerField(null=True)
    request_size = IntegerField(default=0)
    response_size = IntegerField(default=0)
    status_code = IntegerField(null=True)
    
    # Token metrics for performance analytics
    prompt_tokens = IntegerField(null=True)
    completion_tokens = IntegerField(null=True) 
    total_tokens = IntegerField(null=True)
    
    # Request status tracking
    completed_at = DateTimeField(null=True)  # NULL = inflight, NOT NULL = completed
    
    # Payload storage (blob keys for side-car storage)
    request_body_key = CharField(max_length=64, null=True)  # SHA256 hash key
    response_body_key = CharField(max_length=64, null=True)  # SHA256 hash key
    
    # Error tracking
    error_message = TextField(null=True)
    
    @property
    def request_body(self) -> bytes:
        """Get request body from blob storage"""
        if not self.request_body_key:
            return None
        return get_blob_storage().retrieve(self.request_body_key, self.id)
    
    @property
    def response_body(self) -> bytes:
        """Get response body from blob storage"""
        if not self.response_body_key:
            return None
        return get_blob_storage().retrieve(self.response_body_key, self.id)
    
    def set_request_body(self, data: bytes) -> None:
        """Store request body in blob storage with record_id sharding"""
        if data:
            self.request_body_key = get_blob_storage().store(data, "application/json", self.id)
        else:
            self.request_body_key = None
    
    def set_response_body(self, data: bytes) -> None:
        """Store response body in blob storage with record_id sharding"""
        if data:
            self.response_body_key = get_blob_storage().store(data, "application/json", self.id)
        else:
            self.response_body_key = None
    
    def delete_blobs(self) -> None:
        """Delete associated blob data"""
        storage = get_blob_storage()
        if self.request_body_key:
            storage.delete(self.request_body_key, self.id)
        if self.response_body_key:
            storage.delete(self.response_body_key, self.id)
    
    class Meta:
        indexes = (
            # Primary query indexes
            (('timestamp',), False),
            (('service_type', 'timestamp'), False),
            (('status_code', 'timestamp'), False),
            (('completed_at',), False),  # For inflight queries
            (('prompt_tokens', 'duration_ms'), False),  # For performance analytics
            # Optimizations for 1M+ records
            (('service_type',), False),  # For fast service type counts
            (('original_model', 'timestamp'), False),  # Model-specific queries
            (('mapped_model', 'timestamp'), False),  # Model performance tracking
            (('auth_user', 'timestamp'), False),  # User-specific analytics
            # Composite index for performance API
            (('completed_at', 'prompt_tokens', 'duration_ms'), False),
        )

def init_database():
    """Initialize database and create tables"""
    try:
        if not db.is_connection_usable():
            db.connect()
        db.create_tables([RequestLog], safe=True)
        logger.info(f"Database initialized at {DB_PATH}")
        
        # Only run cleanup on startup if explicitly enabled (not for 1M+ record scenarios)
        if os.getenv("CLEANUP_ON_STARTUP", "true").lower() in ("1", "true", "yes"):
            cleanup_old_logs()
            vacuum_database()
        else:
            logger.info("Startup cleanup disabled (CLEANUP_ON_STARTUP=false)")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def cleanup_old_logs():
    """Remove logs older than MAX_AGE_DAYS and their associated blobs"""
    try:
        cutoff_date = datetime.now() - timedelta(days=MAX_AGE_DAYS)
        
        # Get logs to delete and clean up their blobs first
        old_logs = RequestLog.select().where(RequestLog.timestamp < cutoff_date)
        blob_deletion_count = 0
        
        for log in old_logs:
            try:
                log.delete_blobs()
                blob_deletion_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete blobs for log {log.id}: {e}")
        
        # Delete the log entries
        deleted_count = RequestLog.delete().where(RequestLog.timestamp < cutoff_date).execute()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old log entries and {blob_deletion_count} blob sets (older than {MAX_AGE_DAYS} days)")
        
        # Also cleanup orphaned blobs
        try:
            blob_storage = get_blob_storage()
            orphaned_count = blob_storage.cleanup_old(MAX_AGE_DAYS)
            if orphaned_count > 0:
                logger.info(f"Cleaned up {orphaned_count} orphaned blob files")
        except Exception as e:
            logger.warning(f"Failed to cleanup orphaned blobs: {e}")
        
    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")

def vacuum_database():
    """Run VACUUM to reclaim space after cleanup"""
    try:
        db.execute_sql("VACUUM")
        logger.info("Database vacuum completed")
    except Exception as e:
        logger.error(f"Failed to vacuum database: {e}")

def get_recent_logs(limit=100, service_type=None):
    """Get recent request logs"""
    query = RequestLog.select().order_by(RequestLog.timestamp.desc()).limit(limit)
    
    if service_type:
        query = query.where(RequestLog.service_type == service_type)
    
    return list(query)

def get_inflight_requests():
    """Get currently inflight (incomplete) requests"""
    try:
        return list(RequestLog.select().where(RequestLog.completed_at.is_null()).order_by(RequestLog.timestamp.desc()))
    except Exception as e:
        logger.error(f"Failed to get inflight requests: {e}")
        return []

def get_log_stats():
    """Get basic statistics about the logs (optimized for 1M+ records)"""
    try:
        # Use efficient single query with aggregation instead of multiple COUNT queries
        yesterday = datetime.now() - timedelta(days=1)
        
        # Get all stats in one query using SQL aggregation
        # Use the RequestLog model's database connection to ensure test isolation
        model_db = RequestLog._meta.database
        stats_query = model_db.execute_sql("""
            SELECT 
                COUNT(*) as total_requests,
                SUM(CASE WHEN service_type = 'openai' THEN 1 ELSE 0 END) as openai_requests,
                SUM(CASE WHEN service_type = 'ollama' THEN 1 ELSE 0 END) as ollama_requests,
                SUM(CASE WHEN timestamp > ? THEN 1 ELSE 0 END) as recent_requests,
                SUM(CASE WHEN completed_at IS NULL THEN 1 ELSE 0 END) as inflight_requests
            FROM requestlog
        """, (yesterday,))
        
        row = stats_query.fetchone()
        
        return {
            'total_requests': row[0] or 0,
            'openai_requests': row[1] or 0,
            'ollama_requests': row[2] or 0,
            'recent_requests': row[3] or 0,
            'inflight_requests': row[4] or 0
        }
    except Exception as e:
        logger.error(f"Failed to get log stats: {e}")
        return {
            'total_requests': 0,
            'openai_requests': 0,
            'ollama_requests': 0,
            'recent_requests': 0,
            'inflight_requests': 0
        }