# Risk Assessment API

FastAPI server for multi-miner risk assessment and validation.

## Quick Start

```bash
# Start the API server
python scripts/start_api.py

# With custom port
python scripts/start_api.py --port 8001

# With auto-reload for development
python scripts/start_api.py --reload
```

## Configuration

Set environment variables in `.env`:

```bash
# ClickHouse (per network via get_connection_params)
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=miner

# API
API_HOST=0.0.0.0
API_PORT=8000

# Repository
GITHUB_URL=https://github.com/your-org/risk-assessment
COMMIT_HASH=$(git rev-parse HEAD)
```

## Network Support

Network is **always** a required query parameter. The API dynamically creates ClickHouse connections based on the network parameter using `get_connection_params(network)`.

Example networks: `torus`, `bitcoin`, `ethereum`

## Endpoints

All endpoints require `network` query parameter.

### Miner Submissions
- `POST /api/v1/submissions` - Submit miner scores for validation
- `GET /api/v1/submissions/{miner_id}/latest` - Get latest submission from a miner
- `GET /api/v1/scores/{miner_id}/latest` - Get latest validation scores for a miner
- `GET /api/v1/scores/rankings` - Get ranked list of miners by final score

## Testing

```bash
# Submit miner scores
curl -X POST "http://localhost:8000/api/v1/submissions?network=torus" \
  -H "Content-Type: application/json" \
  -d @submission.json

# Get latest validation scores
curl "http://localhost:8000/api/v1/scores/miner_123/latest?network=torus"

# Get miner rankings
curl "http://localhost:8000/api/v1/scores/rankings?network=torus&processing_date=2025-10-31&window_days=195"
```

## API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## How It Works

1. Network parameter in query determines which database to connect to
2. `get_connection_params(network)` returns connection config for that network
3. `ClientFactory` creates connection to the appropriate ClickHouse database
4. API serves miner submissions and validation results from that database