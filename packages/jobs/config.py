import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class CeleryConfig:
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: Optional[str] = None
    timezone: str = "UTC"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list = None
    result_expires: int = 3600
    task_acks_late: bool = True
    worker_prefetch_multiplier: int = 1
    worker_concurrency: int = 4
    
    def __post_init__(self):
        if self.accept_content is None:
            self.accept_content = ['json']
    
    @classmethod
    def from_env(cls) -> 'CeleryConfig':
        return cls(
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            redis_password=os.getenv('REDIS_PASSWORD') or None,
            timezone=os.getenv('CELERY_TIMEZONE', 'UTC'),
            worker_concurrency=int(os.getenv('CELERY_WORKER_CONCURRENCY', '4')),
        )
    
    def get_broker_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_result_backend_url(self) -> str:
        return self.get_broker_url()
    
    def to_celery_config(self) -> dict:
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


@dataclass
class SOTConfig:
    sot_host: str
    sot_port: int
    sot_database: str
    sot_user: str = "default"
    sot_password: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'SOTConfig':
        return cls(
            sot_host=os.getenv('SOT_HOST', 'sot.clickhouse.example.com'),
            sot_port=int(os.getenv('SOT_PORT', '8123')),
            sot_database=os.getenv('SOT_DATABASE', 'sot_production'),
            sot_user=os.getenv('SOT_USER', 'default'),
            sot_password=os.getenv('SOT_PASSWORD'),
        )


@dataclass
class IngestionConfig:
    networks: List[str]
    window_days: int
    retry_attempts: int = 3
    retry_delay: int = 300
    
    @classmethod
    def from_env(cls) -> 'IngestionConfig':
        networks_str = os.getenv('INGESTION_NETWORKS', 'ethereum,bitcoin,torus')
        return cls(
            networks=networks_str.split(','),
            window_days=int(os.getenv('INGESTION_WINDOW_DAYS', '195')),
            retry_attempts=int(os.getenv('INGESTION_RETRY_ATTEMPTS', '3')),
            retry_delay=int(os.getenv('INGESTION_RETRY_DELAY', '300')),
        )