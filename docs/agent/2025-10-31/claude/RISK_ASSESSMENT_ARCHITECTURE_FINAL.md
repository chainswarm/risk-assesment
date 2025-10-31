# Risk Assessment Architecture (Validator Core) - FINAL

**Date**: 2025-10-31  
**Purpose**: Core risk assessment engine for validator - hybrid validation approach  
**Status**: Final Architecture Design

---

## Executive Summary

This document defines the architecture for the **risk assessment engine** - the core component used by the validator in the subnet.

**Key insight from SOT_TIMESERIES_VALIDATION.md**: SOT already has ground truth via [`raw_address_labels`](../../packages/storage/schema/raw_address_labels.sql)!

### Hybrid Validation Approach

✅ **Immediate GT Validation** (for ~10% of alerts with address labels)  
✅ **Behavioral Evolution Validation** (for remaining 90%)  
✅ **Feature Evolution Tracking** (using SOT's 98 time-series features)  

---

## System Context

```
┌────────────────────────────────────────────────────────┐
│        RISK ASSESSMENT ENGINE (THIS PROJECT)           │
│  ┌──────────────────────────────────────────────────┐ │
│  │  1. SOT Data Sync (daily)                        │ │
│  │     - raw_alerts, raw_features, raw_clusters     │ │
│  │     - raw_money_flows, raw_address_labels ⭐     │ │
│  │                                                    │ │
│  │  2. Miner Score Ingestion (via validator)        │ │
│  │     - POST /internal/miner/submit                │ │
│  │                                                    │ │
│  │  3. Multi-Tier Validation                        │ │
│  │     Tier 1: Integrity (0.2 pts)                  │ │
│  │     Tier 2: Behavior (0.3 pts)                   │ │
│  │     Tier 3: Hybrid Validation (0.5 pts)          │ │
│  │       ├─ Address Label GT (~10% coverage)        │ │
│  │       └─ Feature Evolution (~90% coverage)       │ │
│  │                                                    │ │
│  │  4. Results Storage (ClickHouse)                 │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

---

## Component 1: Database Schemas

### New Tables

#### 1. miner_submissions

```sql
CREATE TABLE IF NOT EXISTS miner_submissions (
    submission_id String,
    miner_id String,
    network String,
    processing_date Date,
    window_days UInt16,
    
    alert_id String,
    score Float64,
    model_version String,
    
    submitted_at DateTime DEFAULT now(),
    submission_metadata String DEFAULT '{}'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (network, processing_date, window_days, miner_id, alert_id)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_miner_id ON miner_submissions(miner_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_alert_id ON miner_submissions(alert_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
```

#### 2. feature_evolution_tracking

Tracks SOT's 98 features over time for validation.

```sql
CREATE TABLE IF NOT EXISTS feature_evolution_tracking (
    alert_id String,
    address String,
    network String,
    base_date Date,
    snapshot_date Date,
    window_days UInt16,
    
    degree_delta Int32,
    counterparty_delta Int32,
    volume_delta Decimal128(18),
    tx_delta Int32,
    anomaly_delta Float32,
    structuring_delta Float32,
    
    became_mixer_like Bool DEFAULT false,
    became_dormant Bool DEFAULT false,
    
    evolution_pattern String DEFAULT '',
    pattern_confidence Float32 DEFAULT 0.0,
    
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(base_date)
ORDER BY (network, base_date, alert_id, snapshot_date)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_alert_evolution ON feature_evolution_tracking(alert_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
```

#### 3. miner_validation_results

```sql
CREATE TABLE IF NOT EXISTS miner_validation_results (
    validation_id String,
    miner_id String,
    submission_id String,
    network String,
    processing_date Date,
    window_days UInt16,
    
    tier1_integrity_score Float32,
    tier1_passed Bool,
    
    tier2_behavior_score Float32,
    tier2_traps_detected Array(String),
    
    tier3_gt_score Float32,
    tier3_gt_coverage Float32,
    tier3_auc Float32,
    tier3_brier Float32,
    
    tier3_evolution_score Float32,
    tier3_evolution_validated UInt32,
    
    final_score Float32,
    
    validated_at DateTime DEFAULT now(),
    validation_details String DEFAULT '{}'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(processing_date)
ORDER BY (network, processing_date, window_days, miner_id, final_score)
SETTINGS index_granularity = 8192;

CREATE INDEX IF NOT EXISTS idx_miner_validation ON miner_validation_results(miner_id) 
    TYPE bloom_filter(0.01) GRANULARITY 4;
```

---

## Component 2: Validation Workflow

### Multi-Tier Validation

```
┌──────────────────────────────────────────────────┐
│  VALIDATION SCORING (TOTAL = 1.0)                │
├──────────────────────────────────────────────────┤
│                                                   │
│  Tier 1: Integrity (0.2 pts - Day 0)            │
│  ├─ Schema validation                            │
│  ├─ Completeness check                           │
│  ├─ Score range [0,1]                            │
│  └─ Latency check                                │
│                                                   │
│  Tier 2: Behavior (0.3 pts - Day 0)             │
│  ├─ Pattern traps                                │
│  ├─ Gaming detection                             │
│  └─ Adversarial testing                          │
│                                                   │
│  Tier 3: Hybrid Validation (0.5 pts - T+0/T+30) │
│  ├─ Address Label GT (T+0)                       │
│  │  └─ AUC/Brier for ~10% with labels           │
│  │     Coverage: 0.05-0.15 typically             │
│  │     Weight: coverage * 0.5                    │
│  └─ Feature Evolution (T+30)                     │
│     └─ Track 98 SOT features for ~90%            │
│        Coverage: 0.85-0.95 typically             │
│        Weight: (1-gt_coverage) * 0.5             │
│                                                   │
└──────────────────────────────────────────────────┘
```

### Tier 3: Hybrid Validation

```python
class HybridValidator:
    
    def validate(self, miner_scores, processing_date, window_days):
        
        # Part A: Address Label GT (Immediate)
        gt_score, gt_coverage = self.validate_with_address_labels(
            miner_scores, processing_date, window_days
        )
        
        # Part B: Feature Evolution (30 days)
        evolution_score, evolution_coverage = self.validate_with_feature_evolution(
            miner_scores, processing_date, window_days
        )
        
        # Weighted combination based on coverage
        tier3_score = (gt_coverage * gt_score) + (evolution_coverage * evolution_score)
        
        return {
            'tier3_total': tier3_score,
            'gt_score': gt_score,
            'gt_coverage': gt_coverage,
            'evolution_score': evolution_score,
            'evolution_coverage': evolution_coverage
        }
    
    def validate_with_address_labels(self, miner_scores, processing_date, window_days):
        
        query = f"""
        SELECT 
            a.alert_id,
            a.address,
            CASE 
                WHEN l.risk_level IN ('high', 'critical') THEN 1
                WHEN l.risk_level IN ('low', 'medium') THEN 0
                ELSE NULL
            END as ground_truth_label
        FROM raw_alerts a
        LEFT JOIN raw_address_labels l 
            ON a.address = l.address
            AND a.processing_date = l.processing_date
            AND a.window_days = l.window_days
        WHERE a.processing_date = '{processing_date}'
            AND a.window_days = {window_days}
            AND l.risk_level IS NOT NULL
        """
        
        labeled_alerts = execute_query(query)
        
        if len(labeled_alerts) == 0:
            return 0.0, 0.0
        
        y_true = labeled_alerts['ground_truth_label']
        y_pred = [miner_scores[aid] for aid in labeled_alerts['alert_id']]
        
        auc = roc_auc_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred)
        
        score = (0.6 * auc) + (0.4 * (1 - brier))
        
        total_alerts = get_total_alert_count(processing_date, window_days)
        coverage = len(labeled_alerts) / total_alerts
        
        return score, coverage
    
    def validate_with_feature_evolution(self, miner_scores, processing_date, window_days):
        
        base_date = processing_date
        evolved_date = processing_date + timedelta(days=28)
        
        query = f"""
        WITH 
          baseline AS (
            SELECT 
                f.address,
                f.degree_total,
                f.unique_counterparties,
                f.total_volume_usd,
                f.tx_total_count,
                f.behavioral_anomaly_score,
                f.structuring_score,
                f.is_mixer_like
            FROM raw_features f
            JOIN raw_alerts a ON f.address = a.address
                AND f.processing_date = a.processing_date
                AND f.window_days = a.window_days
            WHERE a.processing_date = '{base_date}'
                AND a.window_days = {window_days}
          ),
          evolved AS (
            SELECT 
                f.address,
                f.degree_total,
                f.unique_counterparties,
                f.total_volume_usd,
                f.tx_total_count,
                f.behavioral_anomaly_score,
                f.structuring_score,
                f.is_mixer_like
            FROM raw_features f
            WHERE f.processing_date = '{evolved_date}'
                AND f.window_days = {window_days}
          )
        SELECT 
            a.alert_id,
            b.address,
            e.degree_total - b.degree_total as degree_delta,
            e.unique_counterparties - b.unique_counterparties as counterparty_delta,
            e.total_volume_usd - b.total_volume_usd as volume_delta,
            e.tx_total_count - b.tx_total_count as tx_delta,
            e.behavioral_anomaly_score - b.behavioral_anomaly_score as anomaly_delta,
            e.structuring_score - b.structuring_score as structuring_delta,
            CASE WHEN e.is_mixer_like AND NOT b.is_mixer_like THEN 1 ELSE 0 END as became_mixer
        FROM baseline b
        JOIN evolved e ON b.address = e.address
        JOIN raw_alerts a ON a.address = b.address
            AND a.processing_date = '{base_date}'
            AND a.window_days = {window_days}
        """
        
        feature_deltas = execute_query(query)
        
        validated_count = 0
        total_score = 0.0
        
        for row in feature_deltas.itertuples():
            miner_score = miner_scores[row.alert_id]
            
            pattern = self.classify_evolution_pattern(row)
            
            if pattern['pattern'] == 'expanding_illicit' and miner_score > 0.7:
                total_score += 0.40
                validated_count += 1
            elif pattern['pattern'] == 'benign_indicators' and miner_score < 0.3:
                total_score += 0.30
                validated_count += 1
        
        avg_score = total_score / len(feature_deltas) if len(feature_deltas) > 0 else 0.0
        
        total_alerts = get_total_alert_count(processing_date, window_days)
        coverage = len(feature_deltas) / total_alerts
        
        return avg_score, coverage
    
    def classify_evolution_pattern(self, deltas):
        
        expanding_illicit = (
            deltas.degree_delta > 10 and
            deltas.counterparty_delta > 5 and
            deltas.structuring_delta > 0.2
        )
        
        going_dark = (
            deltas.tx_delta < -10 and
            deltas.volume_delta < 0
        )
        
        stable_benign = (
            abs(deltas.degree_delta) < 3 and
            abs(deltas.anomaly_delta) < 0.1
        )
        
        if expanding_illicit:
            return {'pattern': 'expanding_illicit', 'confidence': 0.85}
        elif going_dark:
            return {'pattern': 'dormant', 'confidence': 0.60}
        elif stable_benign:
            return {'pattern': 'benign_indicators', 'confidence': 0.75}
        else:
            return {'pattern': 'uncertain', 'confidence': 0.50}
```

---

## Component 3: Workflows

### Workflow 1: Daily SOT Data Sync

```bash
python scripts/ingest_data.py \
    --network torus \
    --processing-date 2025-10-31 \
    --days 195
```

**Syncs (Required):**
- raw_alerts (alerts to score)
- raw_features (98 time-series features - includes flow aggregates)
- raw_clusters (cluster evolution tracking + cluster score validation)
- raw_address_labels (ground truth labels)

**Syncs (NOT needed):**
- raw_money_flows (not used in validation - raw_features has flow aggregates)

### Workflow 2: Immediate Validation (T+0)

After miner score submission:

```bash
python scripts/run_immediate_validation.py \
    --processing-date 2025-10-31 \
    --window-days 195
```

**Runs:**
- Tier 1: Integrity
- Tier 2: Behavior
- Tier 3A: Address Label GT (if labels available)

### Workflow 3: Feature Evolution Tracking (Daily)

```bash
python scripts/track_feature_evolution.py \
    --base-date 2025-10-01 \
    --current-date 2025-10-31
```

**Captures:**
- Weekly feature snapshots
- Deltas across 98 features
- Pattern classification

### Workflow 4: Evolution Validation (T+30)

```bash
python scripts/run_evolution_validation.py \
    --base-date 2025-10-01
```

**Runs:**
- Tier 3B: Feature Evolution validation
- Updates final scores

---

## Component 4: Implementation Roadmap

### Phase 1: Database & Core (Week 1)

- Create new table schemas
- Migration script
- Storage utilities

**Files:**
- `packages/storage/schema/miner_submissions.sql`
- `packages/storage/schema/feature_evolution_tracking.sql`
- `packages/storage/schema/miner_validation_results.sql`

### Phase 2: Internal API (Week 2)

- POST /internal/miner/submit
- GET /internal/validation/results
- Request/response models

**Files:**
- `packages/api/routes.py`
- `packages/api/models.py`
- `packages/api/database.py`

### Phase 3: Validation Engine - Tier 1+2 (Week 3)

- Integrity validator
- Behavior validator
- Immediate validation script

**Files:**
- `packages/assessment/tier1_integrity.py`
- `packages/assessment/tier2_behavior.py`
- `scripts/run_immediate_validation.py`

### Phase 4: GT Validation - Tier 3A (Week 4)

- Address label loader
- AUC/Brier computation
- Coverage calculation

**Files:**
- `packages/assessment/tier3a_address_labels.py`
- `packages/assessment/metrics.py`

### Phase 5: Evolution Validation - Tier 3B (Week 5)

- Feature delta extraction (98 features)
- Pattern classification
- Evolution validation script

**Files:**
- `packages/assessment/tier3b_feature_evolution.py`
- `packages/assessment/evolution_patterns.py`
- `scripts/track_feature_evolution.py`
- `scripts/run_evolution_validation.py`

---

## Component 5: Key Design Decisions

### 1. Hybrid Validation Approach

**Decision**: Use BOTH address labels AND feature evolution

**From SOT_TIMESERIES_VALIDATION.md**:
- Address labels provide immediate GT for ~10% of alerts
- Feature evolution provides validation for ~90% of alerts
- Combined coverage approaching 100%

**Benefits**:
- Immediate validation where labels exist
- Behavioral validation for the rest
- Best of both worlds

### 2. Coverage-Weighted Scoring

**Decision**: Weight by coverage

```python
tier3_score = (gt_coverage * gt_score) + (evolution_coverage * evolution_score)

# Example:
# GT: 10% coverage, 0.85 score → 0.10 * 0.85 = 0.085
# Evolution: 90% coverage, 0.75 score → 0.90 * 0.75 = 0.675
# Total: 0.085 + 0.675 = 0.76 (out of 1.0)
```

### 3. Feature Delta Analysis

**Decision**: Use SOT's 98 time-series features

**Rationale**:
- Richest data available
- Objective, quantitative
- No external dependencies

**Key features tracked**:
- Network: degree_total, unique_counterparties, khop1_count
- Activity: total_volume_usd, tx_total_count, activity_days
- Behavioral: behavioral_anomaly_score, structuring_score
- Classification: is_mixer_like, is_dormant_reactivated

### 4. Pattern Classification

**Three patterns**:
- expanding_illicit (confidence: 0.85)
- benign_indicators (confidence: 0.75)
- dormant (confidence: 0.60 - ambiguous)

### 5. Progressive Validation

**Day 0**: Tier 1 + Tier 2 + Tier 3A (address labels)
**Day 30**: Add Tier 3B (feature evolution)

**Allows**:
- Immediate feedback for technical quality
- Immediate GT validation where labels exist
- Full validation after 30 days

---

## Summary

This architecture provides:

✅ **Hybrid validation** - Address labels (10%) + Feature evolution (90%)  
✅ **Immediate GT** - Uses existing raw_address_labels  
✅ **98-feature tracking** - Richest SOT data  
✅ **Coverage-weighted** - Balanced scoring  
✅ **Progressive validation** - Day 0 and Day 30  
✅ **No external dependencies** - All data from SOT  

### Validation Breakdown

```
Total Score = 1.0 points

Tier 1: Integrity (0.2) - Day 0
Tier 2: Behavior (0.3) - Day 0
Tier 3: Hybrid (0.5) - Day 0 + Day 30
  ├─ Address Labels (0.05-0.075) - ~10% coverage, Day 0
  └─ Feature Evolution (0.425-0.45) - ~90% coverage, Day 30
```

### Next Steps

1. **Week 1**: Create database schemas
2. **Week 2**: Build internal API
3. **Week 3**: Build Tier 1+2 validators
4. **Week 4**: Build Tier 3A (address labels)
5. **Week 5**: Build Tier 3B (feature evolution)

---

## Component 6: Data Requirements Analysis

### Required Tables (Minimum for Core Validation)

**1. raw_alerts**
- Purpose: Alerts being scored by miners
- Used in: All tiers
- Size: ~10K alerts/day

**2. raw_features**
- Purpose: 98 time-series features including flow aggregates
- Used in: Tier 3B (feature evolution validation)
- Flow metrics included: `degree_total`, `unique_counterparties`, `total_volume_usd`, `avg_amount_usd`
- Size: ~10K records/day

**3. raw_address_labels**
- Purpose: Ground truth labels
- Used in: Tier 3A (address label GT validation)
- Coverage: ~10% of alerts
- Size: ~1K labels (relatively static)

### Optional Tables (Advanced Validation)

**4. raw_money_flows**
- Purpose: Detailed transaction-level flow analysis
- Used in: Advanced flow pattern detection (Proposal 3 from SOT_TIMESERIES_VALIDATION.md)
- When needed:
  - Transaction-level layering detection
  - Peel chain pattern validation
  - Detailed structuring analysis
  - Evidence trail generation
- When NOT needed:
  - raw_features contains aggregated flow metrics
  - Feature deltas capture flow changes at aggregate level
- Decision: **OPTIONAL** - Include if validator wants transaction-level granularity

**5. raw_clusters**
- Purpose: Cluster-level risk assessment
- Used in: Cluster evolution tracking
- When needed:
  - Validating cluster scores (if miners submit cluster scores)
  - Tracking coordinated activity evolution
- When NOT needed:
  - Core validation works at alert/address level
- Decision: **OPTIONAL** - Include if validating cluster scores

### Recommendation

**Start with minimal set**:
```bash
python scripts/ingest_data.py \
    --network torus \
    --processing-date 2025-10-31 \
    --days 195 \
    --tables raw_alerts,raw_features,raw_address_labels
```

**Add optional tables later** if needed:
```bash
python scripts/ingest_data.py \
    --network torus \
    --processing-date 2025-10-31 \
    --days 195 \
    --tables raw_alerts,raw_features,raw_address_labels,raw_clusters,raw_money_flows
```

### Storage Impact

```
Minimal Set (Required Only):
- raw_alerts: ~100MB/day
- raw_features: ~500MB/day
- raw_address_labels: ~10MB (static)
Total: ~610MB/day

Full Set (All Tables):
- raw_alerts: ~100MB/day
- raw_features: ~500MB/day
- raw_address_labels: ~10MB
- raw_clusters: ~50MB/day
- raw_money_flows: ~2GB/day
Total: ~2.6GB/day

Savings: 4.3x reduction by skipping money_flows
```

### Answer: Do We Need money_flows?

**NO for core validation** - raw_features contains aggregated flow metrics sufficient for feature evolution tracking.

**YES for advanced validation** - If validator wants:
- Transaction-level pattern detection
- Detailed evidence generation
- Flow-specific behavioral signatures

**Recommendation**: Start WITHOUT money_flows, add later if needed.

---

**Ready for code mode implementation!**