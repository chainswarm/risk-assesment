import os
import json
import logging
from celery import Celery
from celery.schedules import crontab
from celery.signals import setup_logging
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


@setup_logging.connect
def setup_loguru(**kwargs):
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.INFO)
    
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


celery_app = Celery('risk-assessment-jobs')


def load_beat_schedule():
    try:
        schedule_path = os.path.join(os.path.dirname(__file__), 'beat_schedule.json')
        with open(schedule_path, 'r') as f:
            schedule = json.load(f)
        
        for task_name, task_config in schedule.items():
            if 'args' in task_config and isinstance(task_config['args'], list):
                task_config['args'] = tuple(task_config['args'])
            
            if 'schedule' in task_config and isinstance(task_config['schedule'], str):
                cron_parts = task_config['schedule'].split()
                if len(cron_parts) == 5:
                    minute, hour, day, month, day_of_week = cron_parts
                    task_config['schedule'] = crontab(
                        minute=minute,
                        hour=hour,
                        day_of_month=day,
                        month_of_year=month,
                        day_of_week=day_of_week
                    )
        
        return schedule
    except Exception as e:
        logger.error("Failed to load beat_schedule.json", error=str(e))
        return {}


beat_schedule = load_beat_schedule()

celery_app.config_from_object({
    'broker_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    'result_backend': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    'timezone': 'UTC',
    'result_expires': 3600,
    'task_acks_late': True,
    'worker_prefetch_multiplier': 1,
    'beat_schedule': beat_schedule,
    'worker_hijack_root_logger': False,
    'worker_log_color': False,
})

celery_app.autodiscover_tasks([
    'packages.jobs.tasks',
])


def get_celery_app():
    return celery_app


__all__ = ['celery_app', 'get_celery_app']


if __name__ == '__main__':
    import threading
    
    logger.info("Starting Celery for local development")
    
    def run_beat():
        logger.info("Starting Celery Beat scheduler")
        celery_app.start(['beat', '--loglevel=info'])
    
    beat_thread = threading.Thread(target=run_beat)
    beat_thread.daemon = True
    beat_thread.start()
    
    logger.info("Starting Celery Worker")
    celery_app.worker_main(['worker', '--loglevel=info'])