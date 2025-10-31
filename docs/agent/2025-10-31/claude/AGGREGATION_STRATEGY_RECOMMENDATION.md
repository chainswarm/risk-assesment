# Alert Score Aggregation Strategy - Professional Recommendation

## Problem Statement

**Scenario:** One address can have multiple alerts:
```
Address 0xABC:
├─ alert_001 (mixing typology) → miner scores 0.95
├─ alert_002 (layering typology) → miner scores 0.90
└─ alert_003 (structuring typology) → miner scores 0.85

Features tracked ONCE for address:
- degree_delta: +250%
- volume_delta: +500%
- is_mixer_like: true
```

**Question:** How to validate and aggregate scores?

---

## Recommended Solution: Hybrid Approach

### Strategy Overview

**Validate each alert independently, aggregate fairly for final score**

```python
# Step 1: Validate EACH alert independently
for submission in miner_submissions:
    alert_id = submission.alert_id
    score = submission.score
    address = get_address_from_alert(alert_id)
    
    # Validate score against address evolution
    validation_result = validate_score(score, address_evolution)
    store_validation_result(alert_id, validation_result)

# Step 2: Aggregate for final miner score
address_scores = group_by_address(validation_results)
for address, alert_results in address_scores:
    address_avg_score = mean(alert_results)
    
    # Penalty for inconsistency
    if std(alert_results) > 0.15:  # High variance
        consistency_penalty = -0.1
    
final_miner_score = mean(all_address_avg_scores)
```

---

## Detailed Explanation

### Phase 1: Independent Validation (Option A)

**Store EVERY validation result:**

```sql
-- miner_validation_results table
INSERT INTO alert_validation_details (
    miner_id,
    alert_id,
    address,
    submitted_score,
    evolution_validation_score,
    pattern_match,
    created_at
)
```

**Benefits:**
1. **Full audit trail** - See all predictions
2. **Detect gaming** - Inconsistent scoring visible
3. **Transparency** - Every score recorded
4. **Debugging** - Track which alerts scored poorly

**Example Data:**
```
miner_123, alert_001, 0xABC, score=0.95, validation=0.88, pattern=expanding
miner_123, alert_002, 0xABC, score=0.90, validation=0.82, pattern=expanding
miner_123, alert_003, 0xABC, score=0.85, validation=0.79, pattern=expanding
```

### Phase 2: Address-Level Aggregation

**Group by unique address, calculate consistency:**

```python
def aggregate_by_address(alert_validations):
    address_groups = {}
    
    for validation in alert_validations:
        address = validation.address
        if address not in address_groups:
            address_groups[address] = []
        address_groups[address].append(validation.score)
    
    address_scores = []
    for address, scores in address_groups.items():
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Consistency check
        if std_score > 0.15:  # Threshold
            logger.warning(f"Inconsistent scoring for {address}: std={std_score}")
            consistency_penalty = -0.1
        else:
            consistency_penalty = 0.0
        
        final_address_score = avg_score + consistency_penalty
        address_scores.append(final_address_score)
    
    return np.mean(address_scores)
```

### Phase 3: Tier 2 Consistency Penalty

**Add to Tier 2 behavioral validation:**

```python
def tier2_consistency_check(miner_submissions, validations):
    """Check if miner scores same address consistently"""
    
    address_variance = {}
    for alert in miner_submissions:
        address = alert.address
        score = alert.score
        
        if address not in address_variance:
            address_variance[address] = []
        address_variance[address].append(score)
    
    # Calculate consistency score
    consistency_scores = []
    for address, scores in address_variance.items():
        if len(scores) > 1:
            variance = np.var(scores)
            # Low variance = consistent = good
            consistency = 1.0 - min(variance * 4, 1.0)  # Scale to [0,1]
            consistency_scores.append(consistency)
    
    if consistency_scores:
        overall_consistency = np.mean(consistency_scores)
    else:
        overall_consistency = 1.0  # No multi-alert addresses
    
    return overall_consistency
```

---

## Why This Approach Works

### 1. Anti-Gaming Properties

**Gaming Attempt:** Miner submits random scores for same address
```
Address 0xDEF (expanding pattern, should be HIGH):
├─ alert_010 → score 0.95  ✓ Good
├─ alert_011 → score 0.30  ✗ Bad (inconsistent!)
└─ alert_012 → score 0.85  ✓ Good
```

**Detection:**
```python
scores = [0.95, 0.30, 0.85]
std = 0.28  # > 0.15 threshold
penalty = -0.1
address_score = 0.70 - 0.1 = 0.60  # Penalized!
```

**Smart Miner (consistent):**
```
Address 0xDEF:
├─ alert_010 → score 0.92
├─ alert_011 → score 0.90
└─ alert_012 → score 0.93

std = 0.012  # < 0.15 threshold
penalty = 0.0
address_score = 0.92  # No penalty
```

### 2. Fair Weighting

**Problem with pure Option A:**
```
Address 0xABC: 3 alerts → counted 3x
Address 0xDEF: 1 alert → counted 1x
```

**Our solution:**
```python
# Each address averaged first
0xABC: (0.88 + 0.82 + 0.79) / 3 = 0.83  # 1 score
0xDEF: 0.92                              # 1 score

final = (0.83 + 0.92) / 2 = 0.875  # Fair!
```

### 3. Information Preservation

Unlike Option B (early averaging) or Option C (MAX only), we keep:
- All individual validations
- Per-alert pattern matches
- Consistency metrics
- Address-level aggregates

---

## Implementation Plan

### Database Schema Addition

```sql
-- Add to miner_validation_results
ALTER TABLE miner_validation_results ADD COLUMN
    tier2_consistency_score Float64,
    
-- New table for per-alert details
CREATE TABLE IF NOT EXISTS alert_validation_details (
    miner_id String,
    processing_date Date,
    window_days UInt16,
    alert_id String,
    address String,
    submitted_score Float64,
    evolution_validation_score Float64,
    pattern_classification String,
    pattern_match_score Float64,
    validated_at DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (processing_date, window_days, miner_id, alert_id);
```

### Validation Pipeline

```python
# packages/validation/tier3b_evolution.py

def validate(self, miner_submissions, processing_date, window_days):
    """Enhanced validation with per-alert tracking"""
    
    alert_details = []
    
    # Phase 1: Validate each alert
    for submission in miner_submissions:
        alert_id = submission.alert_id
        score = submission.score
        
        # Get address from alert
        address = self._get_address_from_alert(alert_id)
        
        # Get evolution for address
        evolution = self._get_evolution(address, processing_date, window_days)
        
        # Validate score against evolution
        validation_score = self._validate_single_score(score, evolution)
        
        alert_details.append({
            'alert_id': alert_id,
            'address': address,
            'submitted_score': score,
            'validation_score': validation_score,
            'pattern': evolution.pattern_classification
        })
    
    # Phase 2: Aggregate by address
    address_scores = self._aggregate_by_address(alert_details)
    
    # Phase 3: Calculate final with consistency
    final_score = np.mean([s['final_score'] for s in address_scores])
    consistency_score = self._calculate_consistency(alert_details)
    
    return {
        'tier3_evolution_score': final_score,
        'tier2_consistency_score': consistency_score,
        'alert_details': alert_details,
        'address_aggregates': address_scores
    }
```

---

## Comparison Matrix

| Aspect | Option A (Pure) | Option B (Avg First) | Option C (MAX) | **Hybrid (Recommended)** |
|--------|----------------|---------------------|----------------|--------------------------|
| Detect inconsistency | ✓ | ✗ | ✗ | ✓✓ (+ penalty) |
| Fair address weighting | ✗ | ✓ | ✓ | ✓ |
| Full audit trail | ✓ | ✗ | ✗ | ✓ |
| Anti-gaming | Partial | ✗ | Partial | ✓✓ |
| Information loss | None | High | Medium | None |
| Complexity | Low | Medium | Low | Medium |
| **Production Ready** | ✗ | ✗ | ✗ | **✓** |

---

## Expected Behavior Examples

### Example 1: Consistent Good Miner

```python
Submissions:
    0xABC: [0.92, 0.90, 0.93]  # Expanding pattern
    0xDEF: [0.15, 0.12]         # Benign pattern
    0xGHI: [0.85]               # Expanding pattern

Validation:
    0xABC: avg=0.92, std=0.012 → score=0.92, penalty=0.0
    0xDEF: avg=0.14, std=0.015 → score=0.14, penalty=0.0
    0xGHI: avg=0.85, std=0.0   → score=0.85, penalty=0.0

Final: (0.92 + 0.14 + 0.85) / 3 = 0.637
Tier2 consistency: 0.98
```

### Example 2: Inconsistent/Gaming Miner

```python
Submissions:
    0xABC: [0.95, 0.20, 0.90]  # Wild variance!
    0xDEF: [0.80, 0.15]         # Inconsistent
    0xGHI: [0.30]               # Single

Validation:
    0xABC: avg=0.68, std=0.35 → score=0.68, penalty=-0.1 = 0.58
    0xDEF: avg=0.48, std=0.32 → score=0.48, penalty=-0.1 = 0.38
    0xGHI: avg=0.30, std=0.0  → score=0.30, penalty=0.0

Final: (0.58 + 0.38 + 0.30) / 3 = 0.420
Tier2 consistency: 0.45  # Low!
```

---

## Conclusion

**Recommended: Hybrid Approach**

✅ **Validate independently** - Full transparency  
✅ **Aggregate by address** - Fair weighting  
✅ **Penalize inconsistency** - Anti-gaming  
✅ **Store all details** - Debugging & audit  
✅ **Professional quality** - Production-ready  

This approach balances:
- **Accuracy** (each alert validated)
- **Fairness** (addresses weighted equally)
- **Security** (gaming detected and penalized)
- **Transparency** (full audit trail)

**Implementation Effort:** ~2-3 hours to enhance current code with address aggregation and consistency checks.