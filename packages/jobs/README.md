# Celery Jobs Package

Automated task scheduling for the Risk Assessment system using Celery and Redis.

## Overview

This package provides scheduled tasks for daily SOT (Source of Truth) data ingestion with comprehensive validation.

## Features

- **Scheduled Ingestion**: Daily downloads at 1 AM UTC with retry attempts at 3 AM
- **Multi-Network Support**: Ethereum, Bitcoin, Torus (staggered by 15 minutes)
- **Validation Pipeline**: 
  - META.json existence check
  - File presence verification
  - Checksum validation
  - Automatic cleanup on failure
- **Retry Logic**: 3 attempts per task with configurable delays
- **Docker Integration**: Ready-to-deploy with Redis and ClickHouse

## Architecture

```
Celery Beat (Scheduler)
    ↓
Celery Worker (Task Executor)
    ↓
SOT Ingestion Task
    ↓
┌─────────────────────────────────┐
│ 1. Download from SOT            │
│ 2. Validate META.json           │
│ 3. Check files present          │
│ 4. Verify checksums             │
│ 5. Ingest to ClickHouse         │
│ 6. Cleanup temp files           │
└─────────────────────────────────┘
```

## Setup

### Prerequisites

- Redis server running
- ClickHouse databases configured per network
- SOT connection details

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# SOT Connection
SOT_HOST=sot.clickhouse.example.com
SOT_PORT=8123
SOT_DATABASE=sot_production
SOT_USER=default
SOT_PASSWORD=

# Ingestion Settings
INGESTION_NETWORKS=ethereum,bitcoin,torus
INGESTION_WINDOW_DAYS=195
INGESTION_RETRY_ATTEMPTS=3
INGESTION_RETRY_DELAY=300

# Celery Settings
CELERY_TIMEZONE=UTC
CELERY_WORKER_CONCURRENCY=4
```

## Running Locally

### Option 1: Single Process (Development)

```bash
python -m packages.jobs.celery_app
```

This starts both beat (scheduler) and worker in one process.

### Option 2: Separate Processes (Production-like)

Terminal 1 - Worker:
```bash
celery -A packages.jobs.celery_app worker --loglevel=info
```

Terminal 2 - Beat:
```bash
celery -A packages.jobs.celery_app beat --loglevel=info
```

## Running with Docker

```bash
cd ops/risk-assesment
docker-compose up -d
```

This starts:
- `risk-assessment-redis` - Redis broker
- `risk-assessment-celery-worker` - Task executor
- `risk-assessment-celery-beat` - Task scheduler
- `risk-assessment-api` - FastAPI server
- `risk-assessment-clickhouse` - Database

## Schedule Configuration

Edit `beat_schedule.json` to modify task schedules:

```json
{
    "ethereum-daily-sot-ingestion": {
        "task": "packages.jobs.tasks.sot_ingestion_task.sot_ingestion_task",
        "schedule": "0 1 * * *",
        "args": ["ethereum", 195]
    }
}
```

Cron format: `minute hour day month day_of_week`

Examples:
- `0 1 * * *` - Daily at 1:00 AM UTC
- `*/30 * * * *` - Every 30 minutes
- `0 */2 * * *` - Every 2 hours

## Task Details

### SOT Ingestion Task

**Task**: `sot_ingestion_task`  
**Schedule**: Daily at 1 AM (with retries at 3 AM)  
**Purpose**: Download and validate SOT data, ingest into ClickHouse

**Parameters**:
- `network` (str): Network identifier (ethereum, bitcoin, torus)
- `window_days` (int): Data window in days (default: 195)
- `processing_date` (str, optional): Date to process (default: yesterday)

**Workflow**:
1. Download data from SOT to `/tmp/sot_data/{network}/{date}/`
2. Validate META.json exists
3. Check all files listed in META.json are present
4. Verify checksums for each file
5. If validation fails: cleanup and retry (up to 3 attempts)
6. If validation passes: ingest to ClickHouse tables:
   - `raw_alerts`
   - `raw_features`
   - `raw_clusters`
   - `raw_address_labels`
7. Cleanup temporary download directory

**Returns**:
```json
{
    "status": "success",
    "network": "ethereum",
    "processing_date": "2024-10-30",
    "window_days": 195
}
```

## Monitoring

### Check Task Status

```bash
# Celery worker status
celery -A packages.jobs.celery_app inspect active

# Task stats
celery -A packages.jobs.celery_app inspect stats

# Scheduled tasks
celery -A packages.jobs.celery_app inspect scheduled
```

### View Logs

Docker:
```bash
# Worker logs
docker logs risk-assessment-celery-worker -f

# Beat logs
docker logs risk-assessment-celery-beat -f
```

Local:
Check console output or loguru log files.

## Manual Task Execution

Trigger a task manually (useful for testing):

```python
from packages.jobs.tasks.sot_ingestion_task import sot_ingestion_task

# Synchronous execution
result = sot_ingestion_task(
    network='ethereum',
    window_days=195,
    processing_date='2024-10-30'
)

# Asynchronous execution (requires running worker)
task = sot_ingestion_task.delay(
    network='ethereum',
    window_days=195
)
print(f"Task ID: {task.id}")
print(f"Status: {task.status}")
print(f"Result: {task.result}")
```

Or via CLI:

```bash
celery -A packages.jobs.celery_app call \
    packages.jobs.tasks.sot_ingestion_task.sot_ingestion_task \
    --args='["ethereum", 195]'
```

## Validation Details

### META.json Format

Expected structure:
```json
{
    "files": [
        {
            "filename": "raw_alerts.parquet",
            "checksum": "sha256:abc123..."
        },
        {
            "filename": "raw_features.parquet",
            "checksum": "sha256:def456..."
        }
    ]
}
```

### Validation Steps

1. **META.json Exists**: Checks if META.json file is present
2. **Files Present**: Verifies all files listed in META.json exist
3. **Checksums Valid**: Compares SHA-256 checksums for each file
4. **On Failure**: Cleanup download directory and retry

### Retry Behavior

- **Attempts**: 3 total attempts
- **Delay**: 300 seconds (5 minutes) between attempts
- **Cleanup**: Automatic cleanup on each failed attempt
- **Final Failure**: Raises exception after all retries exhausted

## Troubleshooting

### Task Not Running

1. Check Redis is running:
   ```bash
   redis-cli ping
   # Should return: PONG
   ```

2. Check worker is running:
   ```bash
   celery -A packages.jobs.celery_app inspect active
   ```

3. Check beat schedule loaded:
   ```bash
   celery -A packages.jobs.celery_app inspect registered
   ```

### Download Failures

Common causes:
- SOT host unreachable
- Invalid credentials
- Network connectivity issues

Check logs for specific error messages.

### Validation Failures

1. **META.json missing**: SOT may not have finished processing
2. **Checksum mismatch**: Corrupted download, will auto-retry
3. **Missing files**: Incomplete SOT data, will auto-retry

### Ingestion Failures

1. Check ClickHouse connection:
   ```bash
   curl http://localhost:8123/
   # Should return: Ok.
   ```

2. Verify network database exists:
   ```bash
   echo "SHOW DATABASES" | curl -X POST http://localhost:8123/ --data-binary @-
   ```

3. Check table schemas match data

## Adding New Networks

1. Update `INGESTION_NETWORKS` in `.env`:
   ```bash
   INGESTION_NETWORKS=ethereum,bitcoin,torus,polygon
   ```

2. Add schedule in `beat_schedule.json`:
   ```json
   "polygon-daily-sot-ingestion": {
       "task": "packages.jobs.tasks.sot_ingestion_task.sot_ingestion_task",
       "schedule": "45 1 * * *",
       "args": ["polygon", 195]
   }
   ```

3. Configure network database connection in storage config

4. Restart Celery beat and worker

## Best Practices

1. **Monitor Logs**: Set up log aggregation for production
2. **Alert on Failures**: Configure alerts for task failures
3. **Resource Limits**: Monitor disk space in `/tmp/sot_data/`
4. **Stagger Schedules**: Offset network tasks to avoid resource contention
5. **Test Locally**: Always test new schedules locally first
6. **Backup Beat Schedule**: Version control `beat_schedule.json`

## Security

- Never commit `.env` file with real credentials
- Use environment variables for sensitive data
- Restrict Redis access to local network
- Use TLS for SOT connections in production
- Rotate credentials regularly

## Performance

- **Worker Concurrency**: Adjust `CELERY_WORKER_CONCURRENCY` based on CPU cores
- **Retry Delays**: Increase `INGESTION_RETRY_DELAY` if SOT processing is slow
- **Cleanup**: `/tmp/sot_data/` is automatically cleaned after each task
- **Redis Memory**: Monitor Redis memory usage, increase if needed

## Development

### Adding New Tasks

1. Create task file in `packages/jobs/tasks/`
2. Import in `packages/jobs/tasks/__init__.py`
3. Add schedule in `beat_schedule.json`
4. Test locally before deploying

### Testing

```python
# Unit test validation
from packages.jobs.tasks.validation_utils import SOTDataValidator
from pathlib import Path

validator = SOTDataValidator(Path('/tmp/test'))
assert validator.validate_meta_exists() == False
```

### Debugging

Enable debug logging:
```bash
celery -A packages.jobs.celery_app worker --loglevel=debug
```

## Related Documentation

- [SOT Ingestion](../ingestion/README.md)
- [Storage Layer](../storage/README.md)
- [API Documentation](../api/README.md)
- [Celery Documentation](https://docs.celeryproject.org/)