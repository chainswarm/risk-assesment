"""
Type-safe configuration management for Celery jobs package.
Follows existing WarehouseConfig patterns with Redis-specific settings.
"""

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


@dataclass
class CeleryConfig:
    """
    Type-safe configuration for Celery with Redis broker.
    
    Use CeleryConfig.from_env() to load from environment variables
    with Redis connection parameters and Celery-specific settings.
    """
    # Redis connection
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: Optional[str] = None
    
    # Celery settings
    timezone: str = "UTC"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list = None
    result_expires: int = 3600  # 1 hour
    task_acks_late: bool = True
    worker_prefetch_multiplier: int = 1
    worker_concurrency: int = 4
    
    # Development vs production Redis host selection
    use_production_redis: bool = False
    redis_host_prod: Optional[str] = None
    
    def __post_init__(self):
        """Ensure accept_content is never None for easier usage."""
        if self.accept_content is None:
            self.accept_content = ['json']
    
    @classmethod
    def from_env(cls, load_dotenv_file: bool = True, use_production: Optional[bool] = None) -> 'CeleryConfig':
        """
        Load configuration from environment variables.
        
        Automatically loads .env file if python-dotenv is available and load_dotenv_file=True.
        
        Expected environment variables:
        - REDIS_HOST (default: localhost)
        - REDIS_HOST_PROD (for production environment)
        - REDIS_PORT (default: 6379)
        - REDIS_DB (default: 0)
        - REDIS_PASSWORD (optional)
        - CELERY_TIMEZONE (default: UTC)
        - CELERY_WORKER_CONCURRENCY (default: 4)
        - USE_PRODUCTION_REDIS (default: False)
        
        Args:
            load_dotenv_file: Whether to load .env file (default: True)
            use_production: Override production Redis usage (None = use env var)
            
        Returns:
            CeleryConfig instance with values from environment
        """
        # Load .env file if available and requested
        if load_dotenv_file and _DOTENV_AVAILABLE:
            load_dotenv()
        
        # Determine Redis host based on production flag
        use_prod = use_production if use_production is not None else os.getenv('USE_PRODUCTION_REDIS', 'false').lower() == 'true'
        redis_host_prod = os.getenv('REDIS_HOST_PROD')
        
        # Select appropriate Redis host
        if use_prod and redis_host_prod:
            redis_host = redis_host_prod
        else:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
        
        return cls(
            redis_host=redis_host,
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            redis_password=os.getenv('REDIS_PASSWORD') or None,
            timezone=os.getenv('CELERY_TIMEZONE', 'UTC'),
            task_serializer='json',
            result_serializer='json',
            accept_content=['json'],
            result_expires=int(os.getenv('CELERY_RESULT_EXPIRES', '3600')),
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            worker_concurrency=int(os.getenv('CELERY_WORKER_CONCURRENCY', '4')),
            use_production_redis=use_prod,
            redis_host_prod=redis_host_prod
        )
    
    def get_broker_url(self) -> str:
        """
        Get Redis broker URL for Celery.
        
        Returns:
            Redis URL string in format: redis://[:password@]host:port/db
        """
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_result_backend_url(self) -> str:
        """
        Get Redis result backend URL for Celery.
        
        Returns:
            Redis URL string for result backend
        """
        return self.get_broker_url()  # Use same Redis instance for results
    
    def to_celery_config(self) -> dict:
        """
        Convert to dictionary format expected by Celery configuration.
        
        Returns:
            Dictionary with Celery configuration parameters
        """
        return {
            'broker_url': self.get_broker_url(),
            'result_backend': self.get_result_backend_url(),
            'timezone': self.timezone,
            'task_serializer': self.task_serializer,
            'result_serializer': self.result_serializer,
            'accept_content': self.accept_content,
            'result_expires': self.result_expires,
            'task_acks_late': self.task_acks_late,
            'worker_prefetch_multiplier': self.worker_prefetch_multiplier,
            'worker_concurrency': self.worker_concurrency,
            'task_routes': {
                'packages.jobs.tasks.*': {'queue': 'default'},
            },
            'task_default_queue': 'default',
            'task_default_exchange': 'default',
            'task_default_exchange_type': 'direct',
            'task_default_routing_key': 'default',
        }