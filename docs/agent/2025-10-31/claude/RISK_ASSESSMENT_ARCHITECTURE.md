# Risk Assessment System Architecture

**Date**: 2025-10-31  
**Purpose**: Complete architecture for multi-miner risk assessment system  
**Status**: Architecture Design

---

## Executive Summary

This document defines the architecture for a **risk assessment system** where:
- **Validator** syncs data from Source of Truth (SOT) daily as timeseries
- **Multiple miners** submit risk scores independently via API
- **Validator** assesses miner submissions using metrics (AUC, Brier, NDCG, etc.)
- **Frontend** displays miner rankings with metrics (AUC, Brier, NDCG, Score, Model Version, GitHub link, Status)

### Key Architecture Principles

✅ **Multi-tiered workflow** - Data sync → Miner scoring → Assessment → Display  
✅ **A/B Testing Ready** - Multiple miners scored independently  
✅ **Timeseries Storage** - All data stored with processing_date + window_days  
✅ **No Data Migration** - Fresh system, no legacy data  
✅ **Metrics-Driven** - Miners ranked by AUC, Brier, NDCG scores  

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   RISK ASSESSMENT SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐                                                   │
│  │   SOT    │  Daily snapshot data                              │
│  │  (S3)    │  (alerts, features, clusters, money_flows)        │
│  └────┬─────┘                                                   │
│       │                                                          │
│       │ 1. Daily Sync                                           │
│       ▼                                                          │
│  ┌──────────────────────────────────────┐                       │
│  │  VALIDATOR (This Repository)         │                       │
│  │  ┌────────────────────────────────┐  │                       │
│  │  │  ClickHouse Database           │  │                       │
│  │  │  - raw_alerts                  │  │                       │
│  │  │  - raw_features                │  │                       │
│  │  │  - raw_clusters                │  │                       │
│  │  │  - raw_money_flows             │  │                       │
│  │  │  - raw_address_labels          │  │                       │
│  │  └────────────────────────────────┘  │                       │
│  └──────────────────────────────────────┘                       │
│       │                                                          │
│       │ 2. Data available for miners                            │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────┐               │
│  │         MINERS (External Systems)            │               │
│  │  ┌─────────────┐  ┌─────────────┐           │               │
│  │  │  Miner A    │  │  Miner B    │  ...      │               │
│  │  │  - Train ML │  │  - Train ML │           │               │
│  │  │  - Generate │  │  - Generate │           │               │
│  │  │    scores   │  │    scores   │           │               │
│  │  └──────┬──────┘  └──────┬──────┘           │               │
│  │         │                 │                   │               │
│  │         │ 3. POST scores  │                   │               │
│  │         ▼                 ▼                   │               │
│  └─────────────────────────────────────────────┘               │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │  VALIDATOR - Miner API               │                       │
│  │  - POST /miner/submit                │                       │
│  │  - Stores submissions in DB          │                       │
│  └──────────────────────────────────────┘                       │
│             │                                                    │
│             │ 4. Stored submissions                             │
│             ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │  ClickHouse - New Tables             │                       │
│  │  - miner_submissions                 │                       │
│  │  - miner_assessments                 │                       │
│  │  - miner_metadata                    │                       │
│  └──────────────────────────────────────┘                       │
│             │                                                    │
│             │ 5. Assessment workflow                            │
│             ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │  VALIDATOR - Assessment Engine       │                       │
│  │  - Compute AUC (vs ground truth)     │                       │
│  │  - Compute Brier score               │                       │
│  │  - Compute NDCG                      │                       │
│  │  - Store in miner_assessments        │                       │
│  └──────────────────────────────────────┘                       │
│             │                                                    │
│             │ 6. Query scores                                   │
│             ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │  VALIDATOR - Public API              │                       │
│  │  - GET /miners/scores                │                       │
│  │  - Returns rankings + metrics        │                       │
│  └──────────────────────────────────────┘                       │
│             │                                                    │
│             │ 7. Display                                        │
│             ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │  FRONTEND                             │                       │
│  │  - Miner leaderboard                 │                       │
│  │  - AUC, Brier, NDCG, Score           │                       │
│  │  - Model Version, GitHub, Status     │                       │
│  └──────────────────────────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Database Schemas

### New Tables for Miner System

#### 1. miner_submissions

Stores raw score submissions from miners.

```sql
CREATE TABLE IF NOT EXISTS miner_submissions (
    submission_id String,                    -- UUID for submission
    miner_id String,                         -- Identifier for miner
    processing_date Date,                    -- Date of data processed
    window_days UInt16,                      -- Window size used
    
    alert_id String,                         -- Alert being scored
    score Float64,                           -- Miner's predicted score [0,1]
    model_version String,                    -- Miner's model version
    github_url String DEFAULT '',            -- Miner's code repository
    
    submitted_at DateTime DEFAULT now(),    -- When submitted
    submission_metadata String DEFAULT '{}'  -- JSON metadata from miner
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (processing_date, window_days, miner_id, alert_id)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_miner_id ON miner_submissions(miner_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_submission_id ON miner_submissions(submission_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_alert_id ON miner_submissions(alert_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
```

#### 2. miner_assessments

Stores computed assessment metrics for each miner.

```sql
CREATE TABLE IF NOT EXISTS miner_assessments (
    assessment_id String,                    -- UUID for assessment
    miner_id String,                         -- Identifier for miner
    submission_id String,                    -- Reference to submission
    processing_date Date,                    -- Date of assessment
    window_days UInt16,                      -- Window size
    
    -- Core Metrics
    auc_score Float64,                       -- AUC-ROC score
    brier_score Float64,                     -- Brier score
    ndcg_score Float64,                      -- NDCG@K score
    ndcg_k UInt16 DEFAULT 500,              -- K value for NDCG
    
    -- Combined Score
    final_score Float64,                     -- Weighted combination
    
    -- Counts
    total_alerts UInt32,                     -- Total alerts scored
    matched_ground_truth UInt32,             -- Alerts with ground truth
    
    -- Status
    status String DEFAULT 'completed',       -- completed, failed, pending
    error_message String DEFAULT '',         -- Error if failed
    
    assessed_at DateTime DEFAULT now(),
    assessment_metadata String DEFAULT '{}'  -- JSON metadata
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (processing_date, window_days, miner_id, final_score)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_miner_assess ON miner_assessments(miner_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_final_score ON miner_assessments(final_score) 
    TYPE minmax GRANULARITY 4;
```

#### 3. miner_metadata

Stores static information about registered miners.

```sql
CREATE TABLE IF NOT EXISTS miner_metadata (
    miner_id String,                         -- Unique miner identifier
    miner_name String,                       -- Display name
    github_url String,                       -- Code repository
    contact_info String DEFAULT '',          -- Contact information
    model_description String DEFAULT '',     -- Model description
    
    status String DEFAULT 'active',          -- active, inactive, banned
    registered_at DateTime DEFAULT now(),
    last_submission_at DateTime DEFAULT now(),
    
    total_submissions UInt32 DEFAULT 0,
    avg_score Float64 DEFAULT 0.0
) ENGINE = ReplacingMergeTree()
ORDER BY miner_id
SETTINGS index_granularity = 8192;
```

---

## Component 2: API Endpoints

### Miner Submission API

#### POST /miner/submit

Endpoint for miners to submit their scores.

**Request Body:**
```json
{
  "miner_id": "miner_abc123",
  "processing_date": "2025-10-31",
  "window_days": 195,
  "model_version": "v1.2.3",
  "github_url": "https://github.com/miner/model",
  "scores": [
    {
      "alert_id": "alert_001",
      "score": 0.8523
    },
    {
      "alert_id": "alert_002",
      "score": 0.2341
    }
  ],
  "metadata": {
    "training_date": "2025-10-30",
    "feature_count": 42,
    "model_type": "xgboost"
  }
}
```

**Response (Success):**
```json
{
  "submission_id": "uuid-1234-5678",
  "miner_id": "miner_abc123",
  "processing_date": "2025-10-31",
  "scores_received": 10000,
  "status": "accepted",
  "submitted_at": "2025-10-31T10:30:00Z"
}
```

**Response (Error):**
```json
{
  "error": "validation_failed",
  "message": "Score out of range [0,1]",
  "details": {
    "alert_id": "alert_003",
    "invalid_score": 1.5
  }
}
```

### Public Query API

#### GET /miners/scores

Get miner scores and rankings for a processing date.

**Query Parameters:**
- `processing_date` (required): Date to query (YYYY-MM-DD)
- `window_days` (optional): Window size filter
- `limit` (optional): Number of results (default: 100)

**Response:**
```json
{
  "processing_date": "2025-10-31",
  "window_days": 195,
  "total_miners": 15,
  "miners": [
    {
      "rank": 1,
      "miner_id": "miner_abc123",
      "miner_name": "SuperScorer",
      "auc": 0.8923,
      "brier": 0.0823,
      "ndcg": 0.9012,
      "final_score": 0.8919,
      "model_version": "v1.2.3",
      "github_url": "https://github.com/miner/model",
      "status": "active",
      "total_alerts": 10000,
      "matched_ground_truth": 2500
    },
    {
      "rank": 2,
      "miner_id": "miner_def456",
      "auc": 0.8712,
      "brier": 0.0945,
      "ndcg": 0.8856,
      "final_score": 0.8674,
      "model_version": "v2.0.1",
      "github_url": "https://github.com/miner2/model",
      "status": "active"
    }
  ],
  "metadata": {
    "assessed_at": "2025-10-31T12:00:00Z",
    "ground_truth_coverage": 0.25
  }
}
```

#### GET /miners/{miner_id}/history

Get historical performance for a specific miner.

**Response:**
```json
{
  "miner_id": "miner_abc123",
  "miner_name": "SuperScorer",
  "history": [
    {
      "processing_date": "2025-10-31",
      "auc": 0.8923,
      "brier": 0.0823,
      "ndcg": 0.9012,
      "final_score": 0.8919,
      "rank": 1
    },
    {
      "processing_date": "2025-10-30",
      "auc": 0.8856,
      "brier": 0.0867,
      "ndcg": 0.8934,
      "final_score": 0.8852,
      "rank": 2
    }
  ],
  "statistics": {
    "avg_auc": 0.8890,
    "avg_brier": 0.0845,
    "avg_ndcg": 0.8973,
    "avg_rank": 1.5,
    "total_submissions": 30
  }
}
```

---

## Component 3: Workflows

### Workflow 1: Daily Data Sync (SOT → Validator)

**Schedule**: Daily at 00:30 UTC

```bash
# Script: scripts/ingest_data.py
python scripts/ingest_data.py \
    --network torus \
    --processing-date 2025-10-31 \
    --days 195
```

**What it does:**
1. Downloads data from S3 SOT bucket
2. Validates parquet files
3. Ingests into ClickHouse tables:
   - `raw_alerts`
   - `raw_features`
   - `raw_clusters`
   - `raw_money_flows`
   - `raw_address_labels`

**Duration:** ~10 minutes

### Workflow 2: Miner Submission (Miner → Validator)

**Trigger**: After miner completes scoring (anytime after data sync)

**Miner Side:**
```python
# Pseudo-code for miner
import requests

# 1. Miner downloads data from validator (or directly from S3)
# 2. Miner trains/loads model
# 3. Miner generates scores
scores = miner_model.predict(alerts_df)

# 4. Submit to validator
response = requests.post(
    "https://validator.io/miner/submit",
    json={
        "miner_id": "miner_abc123",
        "processing_date": "2025-10-31",
        "window_days": 195,
        "model_version": "v1.2.3",
        "github_url": "https://github.com/miner/model",
        "scores": [
            {"alert_id": aid, "score": float(s)}
            for aid, s in zip(alert_ids, scores)
        ]
    }
)
```

**Validator Side (API Handler):**
```python
# packages/api/routes.py

@router.post("/miner/submit")
async def submit_miner_scores(request: MinerSubmissionRequest):
    # 1. Validate request
    validate_submission(request)
    
    # 2. Generate submission_id
    submission_id = generate_uuid()
    
    # 3. Store in miner_submissions table
    store_submission(submission_id, request)
    
    # 4. Return confirmation
    return {
        "submission_id": submission_id,
        "status": "accepted"
    }
```

### Workflow 3: Assessment (Validator)

**Schedule**: Daily at 02:00 UTC (after miners have submitted)

```bash
# Script: scripts/assess_miners.py
python scripts/assess_miners.py \
    --processing-date 2025-10-31 \
    --window-days 195
```

**What it does:**

```python
# packages/assessment/miner_assessment.py

class MinerAssessment:
    def run(self, processing_date, window_days):
        # 1. Load all miner submissions for date
        submissions = get_miner_submissions(processing_date, window_days)
        
        # 2. Load ground truth labels
        ground_truth = get_ground_truth(processing_date, window_days)
        
        # 3. For each miner
        for miner_id in submissions['miner_id'].unique():
            miner_scores = submissions[submissions['miner_id'] == miner_id]
            
            # 4. Merge with ground truth
            merged = miner_scores.merge(
                ground_truth[['alert_id', 'label']],
                on='alert_id',
                how='inner'
            )
            
            # 5. Compute metrics
            metrics = self.compute_metrics(merged)
            
            # 6. Store assessment
            self.store_assessment(
                miner_id=miner_id,
                processing_date=processing_date,
                metrics=metrics
            )
    
    def compute_metrics(self, merged_df):
        from sklearn.metrics import roc_auc_score, brier_score_loss
        from sklearn.metrics import ndcg_score
        
        y_true = merged_df['label'].values
        y_pred = merged_df['score'].values
        
        # AUC-ROC
        auc = roc_auc_score(y_true, y_pred)
        
        # Brier Score
        brier = brier_score_loss(y_true, y_pred)
        
        # NDCG@500
        ndcg = ndcg_score([y_true], [y_pred], k=500)
        
        # Combined Score (weighted average)
        final_score = (0.4 * auc) + (0.3 * (1 - brier)) + (0.3 * ndcg)
        
        return {
            'auc_score': auc,
            'brier_score': brier,
            'ndcg_score': ndcg,
            'final_score': final_score,
            'total_alerts': len(merged_df),
            'matched_ground_truth': len(merged_df)
        }
```

**Duration:** ~5 minutes

### Workflow 4: Frontend Display

**Frontend queries:**
```bash
curl "https://validator.io/miners/scores?processing_date=2025-10-31"
```

**Displays:**
- Miner leaderboard (ranked by `final_score`)
- Metrics: AUC, Brier, NDCG, Final Score
- Model Version
- GitHub URL
- Status (active/inactive)

---

## Component 4: Ground Truth Strategy

### What is Ground Truth?

Ground truth = confirmed labels for alerts (illicit vs legitimate)

### Sources of Ground Truth

1. **Address Labels** (Already have in `raw_address_labels`)
   - High-risk addresses (mixers, scams, sanctioned)
   - Legitimate addresses (exchanges, wallets)

2. **Pattern Detection Results** (Future)
   - Confirmed money laundering patterns
   - Confirmed legitimate patterns

3. **Manual Review** (Optional)
   - Expert analysts review sample of alerts
   - High-confidence labels added

### Ground Truth Query

```sql
SELECT 
    a.alert_id,
    a.address,
    a.processing_date,
    a.window_days,
    CASE 
        WHEN l.risk_level IN ('high', 'critical') THEN 1
        WHEN l.risk_level IN ('low', 'medium') THEN 0
        ELSE NULL
    END as label
FROM raw_alerts a
LEFT JOIN raw_address_labels l 
    ON a.address = l.address
    AND a.processing_date = l.processing_date
    AND a.window_days = l.window_days
WHERE a.processing_date = '2025-10-31'
    AND a.window_days = 195
    AND l.risk_level IS NOT NULL
```

**Coverage:** Initially ~25% of alerts (those with address labels)

---

## Component 5: Implementation Roadmap

### Phase 1: Database Setup (Week 1)

**Tasks:**
1. Create new table schemas
   - [ ] [`miner_submissions.sql`](../../packages/storage/schema/miner_submissions.sql)
   - [ ] [`miner_assessments.sql`](../../packages/storage/schema/miner_assessments.sql)
   - [ ] [`miner_metadata.sql`](../../packages/storage/schema/miner_metadata.sql)

2. Add migration script
   - [ ] [`scripts/init_miner_tables.py`](../../scripts/init_miner_tables.py)

**Deliverables:**
- SQL schema files
- Migration script
- Verification queries

### Phase 2: Miner Submission API (Week 2)

**Tasks:**
1. Create API models
   - [ ] [`packages/api/models.py`](../../packages/api/models.py) - Add `MinerSubmissionRequest`, `MinerSubmissionResponse`

2. Create API routes
   - [ ] [`packages/api/routes.py`](../../packages/api/routes.py) - Add `POST /miner/submit`

3. Create validation logic
   - [ ] [`packages/api/validation.py`](../../packages/api/validation.py) - Validate submissions

4. Create database writers
   - [ ] [`packages/api/database.py`](../../packages/api/database.py) - Write submissions to DB

**Deliverables:**
- Working submission API
- Validation logic
- Error handling
- API documentation

### Phase 3: Assessment Engine (Week 3)

**Tasks:**
1. Create assessment module
   - [ ] [`packages/assessment/__init__.py`](../../packages/assessment/__init__.py)
   - [ ] [`packages/assessment/miner_assessment.py`](../../packages/assessment/miner_assessment.py)
   - [ ] [`packages/assessment/metrics.py`](../../packages/assessment/metrics.py)

2. Create ground truth loader
   - [ ] [`packages/assessment/ground_truth.py`](../../packages/assessment/ground_truth.py)

3. Create assessment script
   - [ ] [`scripts/assess_miners.py`](../../scripts/assess_miners.py)

**Deliverables:**
- Assessment engine
- Metric computation (AUC, Brier, NDCG)
- Batch assessment script

### Phase 4: Public Query API (Week 4)

**Tasks:**
1. Add query endpoints
   - [ ] [`packages/api/routes.py`](../../packages/api/routes.py):
     - `GET /miners/scores`
     - `GET /miners/{miner_id}/history`
     - `GET /miners/leaderboard`

2. Create response models
   - [ ] [`packages/api/models.py`](../../packages/api/models.py) - Add response schemas

3. Create database queries
   - [ ] [`packages/api/database.py`](../../packages/api/database.py) - Add query functions

**Deliverables:**
- Public query API
- Response formatting
- Leaderboard endpoint

### Phase 5: Testing & Documentation (Week 5)

**Tasks:**
1. Create example miner client
   - [ ] [`scripts/examples/example_miner_submission.py`](../../scripts/examples/example_miner_submission.py)

2. Integration testing
   - [ ] Test full workflow: sync → submit → assess → query

3. Documentation
   - [ ] API documentation
   - [ ] Miner integration guide
   - [ ] Frontend integration guide

**Deliverables:**
- Example code
- Integration tests
- Documentation

---

## Component 6: Example Usage

### For Miners

```python
# example_miner_submission.py
import requests
import pandas as pd
from datetime import datetime

# 1. Download data (or use local copy)
processing_date = "2025-10-31"
window_days = 195

# 2. Load your model
model = load_your_trained_model()

# 3. Get alerts to score
alerts_df = get_alerts_from_validator(processing_date, window_days)

# 4. Generate scores
scores = model.predict(alerts_df)

# 5. Prepare submission
submission = {
    "miner_id": "miner_your_id",
    "processing_date": processing_date,
    "window_days": window_days,
    "model_version": "v1.0.0",
    "github_url": "https://github.com/you/model",
    "scores": [
        {
            "alert_id": str(row['alert_id']),
            "score": float(score)
        }
        for idx, (row, score) in enumerate(zip(
            alerts_df.itertuples(), scores
        ))
    ],
    "metadata": {
        "feature_count": model.n_features_,
        "training_date": "2025-10-30"
    }
}

# 6. Submit
response = requests.post(
    "https://validator.io/miner/submit",
    json=submission
)

print(f"Submission ID: {response.json()['submission_id']}")
```

### For Frontend

```javascript
// Fetch miner leaderboard
async function fetchLeaderboard(processingDate) {
    const response = await fetch(
        `https://validator.io/miners/scores?processing_date=${processingDate}`
    );
    const data = await response.json();
    
    // Display in table
    data.miners.forEach(miner => {
        console.log({
            rank: miner.rank,
            name: miner.miner_name,
            auc: miner.auc,
            brier: miner.brier,
            ndcg: miner.ndcg,
            score: miner.final_score,
            version: miner.model_version,
            github: miner.github_url,
            status: miner.status
        });
    });
}

fetchLeaderboard('2025-10-31');
```

---

## Component 7: Key Design Decisions

### 1. No Data Migration
- **Decision**: Always start from fresh data (no legacy migrations)
- **Rationale**: Cleaner architecture, avoid migration complexity
- **Impact**: All data includes `processing_date` and `window_days` for timeseries

### 2. Miner Submissions via API
- **Decision**: Miners POST scores to validator API
- **Rationale**: Centralized collection, easier validation
- **Alternative Considered**: Miners publish to S3, validator polls
- **Why Rejected**: More complex, no immediate validation feedback

### 3. Assessment Timing
- **Decision**: Run assessment daily after miner submission window
- **Rationale**: Batch processing more efficient than real-time
- **Impact**: Miners see results next day

### 4. Score Weighting
- **Decision**: Weighted average of AUC (40%), Brier (30%), NDCG (30%)
- **Rationale**: Balance precision (AUC), calibration (Brier), ranking (NDCG)
- **Tunable**: Weights can be adjusted based on business priorities

### 5. Ground Truth Coverage
- **Decision**: Start with address labels (~25% coverage)
- **Rationale**: Available immediately, high quality
- **Future**: Add pattern detection, manual review

---

## Summary

This architecture provides:

✅ **Complete workflow** - SOT sync → Miner scoring → Assessment → Display  
✅ **Scalable** - Handles multiple miners easily  
✅ **Metrics-driven** - AUC, Brier, NDCG for ranking  
✅ **API-first** - Clean interfaces for miners and frontend  
✅ **Timeseries** - All data stored with date + window  
✅ **No migrations** - Fresh system design  

### Next Steps

1. **Week 1**: Create database schemas
2. **Week 2**: Build submission API
3. **Week 3**: Build assessment engine
4. **Week 4**: Build query API
5. **Week 5**: Test and document

**Ready to proceed with implementation!**