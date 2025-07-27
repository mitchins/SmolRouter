import os
import logging
from datetime import datetime, timedelta
from peewee import *

logger = logging.getLogger("model-rerouter")

# Database configuration
DB_PATH = os.getenv("DB_PATH", "requests.db")
MAX_AGE_DAYS = int(os.getenv("MAX_LOG_AGE_DAYS", "7"))

# Initialize database
db = SqliteDatabase(DB_PATH)

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
    
    # Model mapping
    original_model = CharField(max_length=200, null=True)
    mapped_model = CharField(max_length=200, null=True)
    
    # Performance metrics
    duration_ms = IntegerField(null=True)
    request_size = IntegerField(default=0)
    response_size = IntegerField(default=0)
    status_code = IntegerField(null=True)
    
    # Payload storage (large blobs)
    request_body = BlobField(null=True)
    response_body = BlobField(null=True)
    
    # Error tracking
    error_message = TextField(null=True)
    
    class Meta:
        indexes = (
            # Index for common queries
            (('timestamp',), False),
            (('service_type', 'timestamp'), False),
            (('status_code', 'timestamp'), False),
        )

def init_database():
    """Initialize database and create tables"""
    try:
        if not db.is_connection_usable():
            db.connect()
        db.create_tables([RequestLog], safe=True)
        logger.info(f"Database initialized at {DB_PATH}")
        
        # Run cleanup on startup
        cleanup_old_logs()
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def cleanup_old_logs():
    """Remove logs older than MAX_AGE_DAYS"""
    try:
        cutoff_date = datetime.now() - timedelta(days=MAX_AGE_DAYS)
        deleted_count = RequestLog.delete().where(RequestLog.timestamp < cutoff_date).execute()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old log entries (older than {MAX_AGE_DAYS} days)")
        
    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")

def get_recent_logs(limit=100, service_type=None):
    """Get recent request logs"""
    query = RequestLog.select().order_by(RequestLog.timestamp.desc()).limit(limit)
    
    if service_type:
        query = query.where(RequestLog.service_type == service_type)
    
    return list(query)

def get_log_stats():
    """Get basic statistics about the logs"""
    try:
        total_requests = RequestLog.select().count()
        
        # Count by service type
        openai_count = RequestLog.select().where(RequestLog.service_type == 'openai').count()
        ollama_count = RequestLog.select().where(RequestLog.service_type == 'ollama').count()
        
        # Recent activity (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent_count = RequestLog.select().where(RequestLog.timestamp > yesterday).count()
        
        return {
            'total_requests': total_requests,
            'openai_requests': openai_count,
            'ollama_requests': ollama_count,
            'recent_requests': recent_count
        }
    except Exception as e:
        logger.error(f"Failed to get log stats: {e}")
        return {
            'total_requests': 0,
            'openai_requests': 0,
            'ollama_requests': 0,
            'recent_requests': 0
        }