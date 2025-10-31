# Risk Assessment Architecture (Validator Core)

**Date**: 2025-10-31  
**Purpose**: Core risk assessment engine for validator - behavioral evolution validation  
**Status**: Architecture Design

---

## Executive Summary

This document defines the architecture for the **risk assessment engine** - the core component used by the validator in the subnet:

- **Validator** syncs data from SOT daily as timeseries to ClickHouse
- **Validator** ingests miner scores via internal API (no direct miner access)
- **Assessment engine** validates miners using behavioral evolution tracking (NOT traditional GT)
- **Results** stored in ClickHouse for validator's use

### Key Principles

✅ **Inner layer** - Wrapped by validator in subnet project  
✅ **Behavioral validation** - Evolution tracking, fingerprinting, adversarial tests  
✅ **No external auth** - Validator calls API on behalf of miners  
✅ **Timeseries storage** - All data with processing_date + window_days  
✅ **No frontend** - Results consumed by validator/subnet system  

---

## System Context

```
┌────────────────────────────────────────────────────────┐
│             SUBNET ARCHITECTURE (OUTER)                │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Validator (Separate Subnet Project)             │ │
│  │  - Communicates with miners                      │ │
│  │  - Collects scores from miners                   │ │
│  │  - Calls risk-assessment engine (this project)   │ │
│  └────────────────┬─────────────────────────────────┘ │
│                   │                                    │
│                   ▼                                    │
│  ┌──────────────────────────────────────────────────┐ │
│  │  RISK ASSESSMENT ENGINE (THIS PROJECT - INNER)   │ │
│  │  ┌────────────────────────────────────────────┐  │ │
│  │  │  1. SOT Data Sync                           │  │ │
│  │  │  2. Miner Score Ingestion (via validator)   │  │ │
│  │  │  3. Behavioral Evolution Assessment         │  │ │
│  │  │  4. Results Storage                         │  │ │
│  │  └────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

**This project provides:**
- Daily SOT data sync
- Miner score ingestion API (internal only)
- Behavioral evolution tracking
- Multi-tier validation (Integrity, Behavior, Evolution)
- ClickHouse storage

**This project does NOT provide:**
- Miner-to-validator communication (subnet handles this)
- Authentication (validator is trusted caller)
- Frontend/UI (separate system)
- Traditional ground truth comparison (uses behavioral evolution instead)

---

## Component 1: Database Schemas

### New Tables for Assessment

#### 1. miner_submissions

Stores miner score submissions (ingested by validator on behalf of miners).

```sql
CREATE TABLE IF NOT EXISTS miner_submissions (
    submission_id String,
    miner_id String,
    processing_date Date,
    window_days UInt16,
    
    alert_id String,
    score Float64,
    model_version String,
    
    submitted_at DateTime DEFAULT now(),
    submission_metadata String DEFAULT '{}'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (processing_date, window_days, miner_id, alert_id)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_miner_id ON miner_submissions(miner_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_alert_id ON miner_submissions(alert_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
```

#### 2. behavioral_evolution_tracking

Tracks how alerts evolve over time for validation.

```sql
CREATE TABLE IF NOT EXISTS behavioral_evolution_tracking (
    alert_id String,
    address String,
    base_date Date,
    snapshot_date Date,
    window_days UInt16,
    
    cycle_size UInt32,
    cycle_depth UInt16,
    new_addresses UInt32,
    
    tx_count UInt32,
    volume_usd Decimal128(18),
    unique_counterparties UInt32,
    
    mixing_events UInt16,
    structuring_score Float32,
    dispersion_rate Float32,
    
    evolution_pattern String DEFAULT '',
    pattern_confidence Float32 DEFAULT 0.0,
    
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(base_date)
ORDER BY (base_date, alert_id, snapshot_date)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_alert_evolution ON behavioral_evolution_tracking(alert_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_pattern ON behavioral_evolution_tracking(evolution_pattern) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
```

#### 3. miner_validation_results

Stores multi-tier validation results.

```sql
CREATE TABLE IF NOT EXISTS miner_validation_results (
    validation_id String,
    miner_id String,
    submission_id String,
    processing_date Date,
    window_days UInt16,
    
    tier1_integrity_score Float32,
    tier1_passed Bool,
    tier1_details String DEFAULT '{}',
    
    tier2_behavior_score Float32,
    tier2_traps_detected Array(String),
    tier2_details String DEFAULT '{}',
    
    tier3_evolution_score Float32,
    tier3_alerts_validated UInt32,
    tier3_pattern_matches UInt32,
    tier3_details String DEFAULT '{}',
    
    final_score Float32,
    
    validated_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (processing_date, window_days, miner_id, final_score)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_miner_validation ON miner_validation_results(miner_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_final_score ON miner_validation_results(final_score) 
    TYPE minmax GRANULARITY 4;
```

---

## Component 2: Internal API Endpoints

### Miner Submission API (Internal Only)

Called by validator on behalf of miners.

#### POST /internal/miner/submit

**Request:**
```json
{
  "miner_id": "miner_abc123",
  "processing_date": "2025-10-31",
  "window_days": 195,
  "model_version": "v1.2.3",
  "scores": [
    {"alert_id": "alert_001", "score": 0.8523},
    {"alert_id": "alert_002", "score": 0.2341}
  ]
}
```

**Response:**
```json
{
  "submission_id": "uuid-1234",
  "miner_id": "miner_abc123",
  "scores_received": 10000,
  "status": "accepted"
}
```

### Validation Query API (Internal Only)

#### GET /internal/validation/results

Get validation results for miners.

**Query Parameters:**
- `processing_date` (required)
- `window_days` (optional)
- `miner_id` (optional)

**Response:**
```json
{
  "processing_date": "2025-10-31",
  "window_days": 195,
  "validations": [
    {
      "miner_id": "miner_abc123",
      "tier1_integrity": 0.20,
      "tier2_behavior": 0.28,
      "tier3_evolution": 0.35,
      "final_score": 0.83,
      "tier3_details": {
        "alerts_validated": 2500,
        "expanding_illicit": 185,
        "benign_confirmed": 1820,
        "ambiguous": 495
      }
    }
  ]
}
```

---

## Component 3: Validation Workflow

### Multi-Tier Validation (from ALTERNATIVE_VALIDATION_PROPOSALS.md)

```
┌──────────────────────────────────────────────────┐
│     VALIDATION SCORING (TOTAL = 1.0)             │
├──────────────────────────────────────────────────┤
│                                                   │
│  Tier 1: Integrity (0.2 pts - Day 0)            │
│  ├─ Schema validation                            │
│  ├─ Completeness check                           │
│  ├─ Score range [0,1]                            │
│  ├─ Latency check                                │
│  └─ Determinism                                  │
│                                                   │
│  Tier 2: Behavior (0.3 pts - Day 0)             │
│  ├─ Pattern traps                                │
│  ├─ Gaming detection (variance, plagiarism)      │
│  ├─ Adversarial testing                          │
│  └─ Consistency checks                           │
│                                                   │
│  Tier 3: Behavioral Evolution (0.5 pts - T+30)  │
│  ├─ Network growth tracking                      │
│  ├─ Activity pattern evolution                   │
│  ├─ Mixing intensity changes                     │
│  ├─ Behavior fingerprinting                      │
│  └─ Evolution pattern matching                   │
│                                                   │
└──────────────────────────────────────────────────┘
```

### Tier 1: Integrity Validation (Immediate)

```python
class IntegrityValidator:
    def validate(self, scores_df, alerts_df):
        checks = {
            'completeness': len(scores_df) == len(alerts_df),
            'schema': self.check_required_columns(scores_df),
            'score_range': scores_df['score'].between(0, 1).all(),
            'latency': scores_df['latency_ms'].mean() < 100
        }
        
        if all(checks.values()):
            return {'passed': True, 'score': 0.20}
        return {'passed': False, 'score': 0.00}
```

### Tier 2: Behavior Validation (Immediate)

```python
class BehaviorValidator:
    def validate(self, scores_df, pattern_traps):
        score = 0.30
        traps_detected = []
        
        for trap in pattern_traps:
            actual_score = scores_df[
                scores_df['alert_id'] == trap['alert_id']
            ]['score'].values[0]
            
            if trap['expected'] == 'high' and actual_score < 0.8:
                traps_detected.append(trap['trap_id'])
                score -= 0.05
            elif trap['expected'] == 'low' and actual_score > 0.2:
                traps_detected.append(trap['trap_id'])
                score -= 0.05
        
        variance = scores_df['score'].var()
        if variance < 0.001:
            score -= 0.10
        
        return {
            'score': max(0.0, score),
            'traps_detected': traps_detected
        }
```

### Tier 3: Behavioral Evolution Validation (T+30 days)

```python
class BehavioralEvolutionValidator:
    
    def track_evolution(self, alert_id, address, base_date, days=30):
        evolution = {}
        
        for week in range(1, 5):
            snapshot_date = base_date + timedelta(weeks=week)
            snapshot = self.capture_snapshot(
                address=address,
                snapshot_date=snapshot_date,
                window_days=195
            )
            
            evolution[f'week_{week}'] = {
                'cycle_size': snapshot['connected_addresses'],
                'tx_count': snapshot['transaction_count'],
                'volume_usd': snapshot['total_volume'],
                'mixing_events': snapshot['mixer_interactions'],
                'dispersion_rate': snapshot['fund_dispersion']
            }
        
        return evolution
    
    def classify_pattern(self, evolution):
        week1 = evolution['week_1']
        week4 = evolution['week_4']
        
        expanding_illicit = (
            week4['cycle_size'] > week1['cycle_size'] * 1.5 and
            week4['mixing_events'] > 0 and
            week4['dispersion_rate'] > 0.7
        )
        
        escalating = (
            week4['volume_usd'] > week1['volume_usd'] * 2.0
        )
        
        going_dormant = (
            week4['tx_count'] < week1['tx_count'] * 0.3
        )
        
        if expanding_illicit or escalating:
            return {'pattern': 'illicit_indicators', 'confidence': 0.85}
        elif going_dormant:
            return {'pattern': 'dormant', 'confidence': 0.60}
        else:
            return {'pattern': 'benign_indicators', 'confidence': 0.70}
    
    def validate_prediction(self, miner_score, evolution_pattern):
        if miner_score > 0.7 and evolution_pattern['pattern'] == 'illicit_indicators':
            return {
                'validated': True,
                'score': 0.40,
                'confidence': evolution_pattern['confidence']
            }
        
        elif miner_score < 0.3 and evolution_pattern['pattern'] == 'benign_indicators':
            return {
                'validated': True,
                'score': 0.30,
                'confidence': evolution_pattern['confidence']
            }
        
        elif evolution_pattern['pattern'] == 'dormant':
            return {
                'validated': False,
                'score': 0.0,
                'reason': 'ambiguous_dormancy'
            }
        
        else:
            return {
                'validated': False,
                'score': 0.0,
                'reason': 'evolution_mismatch'
            }
```

---

## Component 4: Workflows

### Workflow 1: Daily SOT Data Sync

**Schedule**: Daily at 00:30 UTC

```bash
python scripts/ingest_data.py \
    --network torus \
    --processing-date 2025-10-31 \
    --days 195
```

**Syncs to ClickHouse:**
- raw_alerts
- raw_features
- raw_clusters
- raw_money_flows
- raw_address_labels

### Workflow 2: Miner Score Ingestion

**Trigger**: Validator calls API after collecting from miners

```python
import requests

response = requests.post(
    "http://localhost:8000/internal/miner/submit",
    json={
        "miner_id": "miner_abc123",
        "processing_date": "2025-10-31",
        "window_days": 195,
        "model_version": "v1.2.3",
        "scores": scores_list
    }
)
```

### Workflow 3: Immediate Validation (T+0)

**Schedule**: Triggered after submission

```bash
python scripts/run_immediate_validation.py \
    --processing-date 2025-10-31 \
    --window-days 195
```

**Runs:**
- Tier 1: Integrity checks
- Tier 2: Behavior validation

**Stores** results in `miner_validation_results` (partial)

### Workflow 4: Evolution Tracking

**Schedule**: Daily background task

```bash
python scripts/track_behavioral_evolution.py \
    --base-date 2025-10-01 \
    --current-date 2025-10-31
```

**What it does:**
1. For each alert from 30 days ago
2. Track weekly snapshots
3. Classify evolution patterns
4. Store in `behavioral_evolution_tracking`

### Workflow 5: Evolution Validation (T+30)

**Schedule**: Daily for alerts that are 30 days old

```bash
python scripts/run_evolution_validation.py \
    --base-date 2025-10-01
```

**What it does:**
1. Load miner submissions from 30 days ago
2. Load evolution patterns for those alerts
3. Compare miner scores vs evolution
4. Update `miner_validation_results` with Tier 3 scores
5. Compute final scores (Tier1 + Tier2 + Tier3)

---

## Component 5: Implementation Roadmap

### Phase 1: Database & Core Infrastructure (Week 1)

**Tasks:**
- Create new table schemas
- Migration script for new tables
- Update storage utilities

**Files:**
- `packages/storage/schema/miner_submissions.sql`
- `packages/storage/schema/behavioral_evolution_tracking.sql`
- `packages/storage/schema/miner_validation_results.sql`
- `scripts/init_assessment_tables.py`

### Phase 2: Internal API (Week 2)

**Tasks:**
- Submission endpoint (POST /internal/miner/submit)
- Query endpoint (GET /internal/validation/results)
- Request/response models
- Database writers

**Files:**
- `packages/api/routes.py` (add internal endpoints)
- `packages/api/models.py` (add new models)
- `packages/api/database.py` (add new queries)

### Phase 3: Validation Engine (Week 3)

**Tasks:**
- Tier 1: Integrity validator
- Tier 2: Behavior validator (pattern traps, gaming detection)
- Immediate validation script

**Files:**
- `packages/assessment/__init__.py`
- `packages/assessment/tier1_integrity.py`
- `packages/assessment/tier2_behavior.py`
- `packages/assessment/pattern_traps.py`
- `scripts/run_immediate_validation.py`

### Phase 4: Behavioral Evolution Tracking (Week 4)

**Tasks:**
- Evolution snapshot capture
- Pattern classification logic
- Background tracking script

**Files:**
- `packages/assessment/behavioral_evolution.py`
- `packages/assessment/evolution_patterns.py`
- `scripts/track_behavioral_evolution.py`

### Phase 5: Evolution Validation (Week 5)

**Tasks:**
- Tier 3: Evolution validator
- Score comparison logic
- Final score computation
- Evolution validation script

**Files:**
- `packages/assessment/tier3_evolution.py`
- `scripts/run_evolution_validation.py`

---

## Component 6: Key Design Decisions

### 1. Behavioral Evolution NOT Traditional GT

**Decision**: Use behavioral evolution tracking instead of AUC/Brier against address labels

**Rationale**: 
- Validates private intelligence (miners may know more than SOT)
- Fully on-chain, no external dependencies
- Works during cold start (no SAR filings needed)

**From**: ALTERNATIVE_VALIDATION_PROPOSALS.md - Proposal 1

### 2. Internal API Only

**Decision**: API called by validator, not directly by miners

**Rationale**:
- This is inner layer (wrapped by subnet validator)
- No auth needed (validator is trusted)
- Cleaner separation of concerns

### 3. Multi-Tier Validation

**Decision**: Three tiers with different timing
- Tier 1 + 2: Immediate (Day 0) - 0.5 points max
- Tier 3: Evolution (Day 30) - 0.5 points max

**Rationale**:
- Immediate feedback for technical quality
- Evolution validation for accuracy
- Miners can be ranked on Day 0 (cold start)

### 4. Evolution Snapshots

**Decision**: Weekly snapshots for 4 weeks (28 days)

**Rationale**:
- Balance between granularity and storage
- 4 weeks enough to see patterns
- Weekly reduces noise vs daily

### 5. Pattern Classification

**Decision**: Three patterns - illicit_indicators, benign_indicators, dormant

**Rationale**:
- Clear categories for validation logic
- Dormant is ambiguous (could be caught or naturally ended)
- Confidence scores reflect uncertainty

---

## Summary

This architecture provides:

✅ **Inner validator core** - Used by subnet validator  
✅ **SOT data sync** - Daily timeseries ingestion  
✅ **Behavioral evolution** - 30-day tracking and validation  
✅ **Multi-tier validation** - Integrity, Behavior, Evolution  
✅ **Internal API** - Called by validator on behalf of miners  
✅ **No traditional GT** - Uses on-chain behavior patterns  
✅ **ClickHouse storage** - All results queryable  

### Next Steps

1. **Week 1**: Create database schemas
2. **Week 2**: Build internal API
3. **Week 3**: Build validation engine (Tier 1+2)
4. **Week 4**: Build evolution tracking
5. **Week 5**: Build evolution validation (Tier 3)

**Ready for code mode implementation!**