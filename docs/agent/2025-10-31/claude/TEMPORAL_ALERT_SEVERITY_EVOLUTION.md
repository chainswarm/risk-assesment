# Temporal Alert Severity Evolution - Day 0 vs Day 30

## The Question

**"At day 0 we have alerts with [critical, high, medium, low] severities. At day 30 we have alerts with [critical, high, medium, low] severities. How does this relate to penalties?"**

---

## Understanding the Timeline

### What We Have

```
Day 0 (T+0): Alert Generated
├─ Alerts detected: alert_001, alert_002, alert_003, alert_004
├─ Severities: [critical, high, medium, low]
├─ Features captured: degree=100, volume=10000
└─ Miner scores these alerts: [0.95, 0.70, 0.50, 0.20]

Day 7 (T+7): Evolution tracking
├─ Features: degree=150, volume=15000
└─ (no new alerts, just tracking)

Day 14 (T+14): Evolution tracking
├─ Features: degree=200, volume=25000
└─ (no new alerts, just tracking)

Day 21 (T+21): Evolution tracking
├─ Features: degree=300, volume=40000
└─ (no new alerts, just tracking)

Day 30 (T+30): Final snapshot + NEW alerts may be detected
├─ Features: degree=400, volume=60000
├─ NEW Alerts: alert_005, alert_006, alert_007
├─ New severities: [critical, high, medium]
└─ Evolution calculated: degree_delta=+300%, volume_delta=+500%
```

### Two Separate Concepts

**Concept 1: Feature Evolution (What we currently validate)**
- Track how address FEATURES change over 30 days
- Use feature deltas to classify patterns
- Validate miner scores against feature evolution

**Concept 2: Alert Severity Evolution (Your question)**
- Compare alert SEVERITIES between T+0 and T+30
- Use this as additional validation signal
- Penalize if miner scores contradict severity escalation

---

## Current Implementation (Feature-Based)

### How We Currently Validate

```python
# Day 0: Miner scores alerts
miner_scores = {
    'alert_001': 0.95,
    'alert_002': 0.70,
    'alert_003': 0.50,
    'alert_004': 0.20
}

# Day 30: We calculate feature evolution
feature_evolution = {
    'degree_delta': +300%,
    'volume_delta': +500%,
    'pattern': 'expanding',
    'expected_score_range': [0.70, 1.00]
}

# Validation: Do miner scores align with feature evolution?
for alert_id, score in miner_scores.items():
    if score in expected_score_range:
        ✓ Good
    else:
        ✗ Bad
```

**Gap:** We're NOT currently using alert severity changes as a signal!

---

## Proposed Enhancement: Alert Severity Evolution Tracking

### New Data to Track

```sql
CREATE TABLE IF NOT EXISTS alert_severity_evolution (
    alert_id String,
    address String,
    base_date Date,
    snapshot_date Date,
    
    -- Day 0 data
    initial_severity String,
    initial_alert_count UInt32,
    initial_critical_count UInt32,
    initial_high_count UInt32,
    initial_medium_count UInt32,
    initial_low_count UInt32,
    
    -- Day 30 data
    final_severity String,
    final_alert_count UInt32,
    final_critical_count UInt32,
    final_high_count UInt32,
    final_medium_count UInt32,
    final_low_count UInt32,
    
    -- Evolution metrics
    severity_escalation Boolean,  -- True if more severe alerts added
    severity_de_escalation Boolean,  -- True if less severe
    new_alerts_count UInt32,
    severity_change_score Float64,
    
    tracked_at DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (base_date, alert_id, snapshot_date);
```

### Severity Evolution Scenarios

#### Scenario 1: Escalating Risk (Expanding Pattern)

```
Address 0xABC:

Day 0 alerts:
- alert_001: medium severity
- alert_002: low severity
Total: 2 alerts, 0 critical, 0 high, 1 medium, 1 low

Day 30 alerts (original + new):
- alert_001: medium severity (same)
- alert_002: low severity (same)
- alert_003: critical severity (NEW!)
- alert_004: high severity (NEW!)
- alert_005: high severity (NEW!)
Total: 5 alerts, 1 critical, 2 high, 1 medium, 1 low

Severity change:
- New alerts: +3
- Critical: 0 → 1 (+1)
- High: 0 → 2 (+2)
- Severity escalation: TRUE
- Change score: +0.8 (high escalation)
```

**Expected Miner Behavior:**
```python
# Miner SHOULD score high at Day 0, even for medium/low alerts
# Because they can predict escalation from features

Day 0 scores (should be HIGH despite low severity):
- alert_001 (medium): 0.85  ✓ Good - predicted escalation
- alert_002 (low): 0.80     ✓ Good - predicted escalation

# If miner just copied severity (wrong):
- alert_001 (medium): 0.50  ✗ Bad - missed escalation signal
- alert_002 (low): 0.20     ✗ Bad - missed escalation signal
```

#### Scenario 2: De-escalating Risk (False Positive)

```
Address 0xDEF:

Day 0 alerts:
- alert_010: critical severity
- alert_011: high severity
Total: 2 alerts, 1 critical, 1 high, 0 medium, 0 low

Day 30 alerts:
- alert_010: critical (same, but no new activity)
- alert_011: high (same, but no new activity)
- No new alerts
Total: 2 alerts, 1 critical, 1 high, 0 medium, 0 low

Feature evolution:
- degree_delta: +5% (minimal growth)
- volume_delta: +10% (minimal growth)
- Pattern: benign (false positive)

Severity change:
- New alerts: 0
- No escalation
- Change score: 0.0 (no change)
```

**Expected Miner Behavior:**
```python
# Miner SHOULD score LOW at Day 0, despite high severity
# Because features suggest false positive

Day 0 scores (should be LOW despite high severity):
- alert_010 (critical): 0.25  ✓ Good - identified false positive
- alert_011 (high): 0.20      ✓ Good - identified false positive

# If miner just copied severity (wrong):
- alert_010 (critical): 0.95  ✗ Bad - missed false positive
- alert_011 (high): 0.70      ✗ Bad - missed false positive
```

#### Scenario 3: Stable Risk (Dormant)

```
Address 0xGHI:

Day 0 alerts:
- alert_020: medium severity
Total: 1 alert, 0 critical, 0 high, 1 medium, 0 low

Day 30 alerts:
- alert_020: medium (same)
Total: 1 alert, 0 critical, 0 high, 1 medium, 0 low

Feature evolution:
- degree_delta: +2%
- volume_delta: +3%
- Pattern: dormant

Severity change:
- New alerts: 0
- No escalation
- Change score: 0.0
```

**Expected Miner Behavior:**
```python
Day 0 scores (should be LOW-MEDIUM):
- alert_020 (medium): 0.35  ✓ Good - dormant pattern

# If miner scored high or low:
- alert_020 (medium): 0.80  ✗ Bad - overestimated
- alert_020 (medium): 0.10  ✗ Bad - underestimated
```

---

## Enhanced Validation with Severity Evolution

### Validation Logic

```python
def validate_with_severity_evolution(miner_score, alert_id, feature_evolution, severity_evolution):
    """Enhanced validation considering both feature and severity evolution"""
    
    # 1. Feature-based validation (current)
    feature_score = validate_feature_evolution(miner_score, feature_evolution)
    
    # 2. Severity-based validation (NEW)
    severity_score = validate_severity_evolution(miner_score, severity_evolution)
    
    # 3. Consistency check
    consistency_score = check_feature_severity_consistency(
        feature_evolution,
        severity_evolution
    )
    
    # 4. Combined score
    final_validation = (
        feature_score * 0.60 +      # Primary: features
        severity_score * 0.30 +     # Secondary: severity
        consistency_score * 0.10    # Tertiary: alignment
    )
    
    return final_validation


def validate_severity_evolution(miner_score, severity_evolution):
    """Validate miner score against severity changes"""
    
    severity_change = severity_evolution.change_score
    new_alerts = severity_evolution.new_alerts_count
    escalation = severity_evolution.severity_escalation
    
    # Expected score adjustment based on severity evolution
    if escalation and severity_change > 0.5:
        # Strong escalation - score should be HIGH
        expected_range = [0.70, 1.00]
    elif escalation and severity_change > 0.2:
        # Moderate escalation - score should be MEDIUM-HIGH
        expected_range = [0.50, 0.80]
    elif not escalation and new_alerts == 0:
        # No new activity - score should reflect initial assessment
        if severity_evolution.initial_severity in ['critical', 'high']:
            # High initial but no escalation - might be false positive
            expected_range = [0.30, 0.70]
        else:
            # Low initial and no escalation - likely benign
            expected_range = [0.00, 0.40]
    else:
        # Unclear pattern
        expected_range = [0.20, 0.80]
    
    # Check if miner score is in expected range
    if expected_range[0] <= miner_score <= expected_range[1]:
        return 1.0  # Perfect alignment
    else:
        # Calculate penalty based on distance from expected range
        if miner_score < expected_range[0]:
            distance = expected_range[0] - miner_score
        else:
            distance = miner_score - expected_range[1]
        
        penalty = min(distance * 2, 1.0)  # Max penalty = 1.0
        return 1.0 - penalty
```

### Penalty Structure

```python
def calculate_severity_evolution_penalty(miner_score, severity_evolution, feature_evolution):
    """Calculate penalty for misalignment with severity evolution"""
    
    # Case 1: Feature shows expanding, severity escalated, but miner scored LOW
    if (feature_evolution.pattern == 'expanding' and 
        severity_evolution.severity_escalation and 
        miner_score < 0.50):
        return -0.15  # Severe penalty - missed obvious escalation
    
    # Case 2: Feature shows benign, no new alerts, but miner scored HIGH
    if (feature_evolution.pattern == 'benign' and 
        severity_evolution.new_alerts_count == 0 and 
        miner_score > 0.70):
        return -0.15  # Severe penalty - false positive amplification
    
    # Case 3: Feature and severity contradict each other
    if (feature_evolution.pattern == 'expanding' and 
        not severity_evolution.severity_escalation):
        # Feature says risky but no new alerts?
        # Miner should be cautious - use features over severity
        if miner_score > 0.80:
            return -0.05  # Minor penalty - too confident
        else:
            return 0.0  # Good - used feature data
    
    # Case 4: Feature shows dormant, but severity escalated
    if (feature_evolution.pattern == 'dormant' and 
        severity_evolution.severity_escalation):
        # New alerts despite low activity - might be detection improvement
        # Miner should moderate
        if miner_score > 0.60 or miner_score < 0.30:
            return -0.10  # Moderate penalty - too extreme
        else:
            return 0.0  # Good - moderate assessment
    
    return 0.0  # No penalty
```

---

## Concrete Examples with Penalties

### Example 1: Perfect Prediction ✓

```
Address 0xAAA:

Day 0:
- Features: degree=100, volume=10000
- Alerts: 1 medium severity
- Miner score: 0.85  ← HIGH despite medium severity

Day 30:
- Features: degree=400, volume=60000 (+300%, +500%)
- Alerts: 4 total (1 original + 3 NEW critical/high)
- Severity escalation: TRUE
- Change score: +0.9

Validation:
- Feature pattern: expanding ✓
- Severity escalated: yes ✓
- Miner scored HIGH despite low initial severity ✓
- Severity penalty: 0.0
- Feature validation: 0.95
- Final score: 0.95 ✓✓✓
```

### Example 2: Copying Severity (Wrong) ✗

```
Same address 0xAAA:

Day 0:
- Features: degree=100, volume=10000
- Alerts: 1 medium severity
- Miner score: 0.50  ← Just copied severity

Day 30:
- Severity escalated to critical
- Features confirmed expanding pattern

Validation:
- Feature pattern: expanding ✓
- Severity escalated: yes ✓
- Miner scored MEDIUM (just copied) ✗
- Severity penalty: -0.15 (missed escalation)
- Feature validation: 0.60
- Final score: 0.60 - 0.15 = 0.45 ✗✗✗
```

### Example 3: False Positive Detection ✓

```
Address 0xBBB:

Day 0:
- Features: degree=50, volume=5000
- Alerts: 1 critical severity
- Miner score: 0.25  ← LOW despite critical severity

Day 30:
- Features: degree=55, volume=5500 (+10%, +10%)
- Alerts: 1 (no new alerts)
- No severity escalation
- Change score: 0.0

Validation:
- Feature pattern: benign ✓
- No severity escalation ✓
- Miner scored LOW despite high initial severity ✓
- Severity penalty: 0.0
- Feature validation: 0.90
- Final score: 0.90 ✓✓✓
```

### Example 4: Over-Confidence in False Positive ✗

```
Same address 0xBBB:

Day 0:
- Features: benign
- Alerts: 1 critical
- Miner score: 0.95  ← Just copied severity

Day 30:
- No escalation
- Features still benign
- Revealed as false positive

Validation:
- Feature pattern: benign✓
- No escalation ✓
- Miner scored HIGH (wrong) ✗
- Severity penalty: -0.15 (amplified false positive)
- Feature validation: 0.30
- Final score: 0.30 - 0.15 = 0.15 ✗✗✗
```

---

## Implementation Approach

### Phase 1: Data Collection (New)

```python
def track_severity_evolution(base_date, snapshot_date, window_days):
    """Track how alert severities change over time"""
    
    # Get alerts at base_date
    base_alerts = get_alerts(base_date, window_days)
    base_severity_dist = calculate_severity_distribution(base_alerts)
    
    # Get alerts at snapshot_date (30 days later)
    snapshot_alerts = get_alerts(snapshot_date, window_days)
    snapshot_severity_dist = calculate_severity_distribution(snapshot_alerts)
    
    # Calculate evolution
    for address in unique_addresses:
        base_dist = base_severity_dist.get(address, {})
        snap_dist = snapshot_severity_dist.get(address, {})
        
        evolution = {
            'address': address,
            'base_date': base_date,
            'snapshot_date': snapshot_date,
            'initial_critical': base_dist.get('critical', 0),
            'initial_high': base_dist.get('high', 0),
            'initial_medium': base_dist.get('medium', 0),
            'initial_low': base_dist.get('low', 0),
            'final_critical': snap_dist.get('critical', 0),
            'final_high': snap_dist.get('high', 0),
            'final_medium': snap_dist.get('medium', 0),
            'final_low': snap_dist.get('low', 0),
            'severity_escalation': check_escalation(base_dist, snap_dist),
            'change_score': calculate_change_score(base_dist, snap_dist)
        }
        
        store_severity_evolution(evolution)
```

### Phase 2: Enhanced Validation

```python
# In tier3b_evolution.py validate() method

def validate(self, miner_submissions, processing_date, window_days):
    """Enhanced validation with severity evolution"""
    
    results = []
    
    for submission in miner_submissions:
        # Current: Feature evolution
        feature_evo = self._get_evolution(submission.address, processing_date, window_days)
        
        # NEW: Severity evolution
        severity_evo = self._get_severity_evolution(submission.address, processing_date, window_days)
        
        # Validate against both
        feature_score = self._validate_feature_evolution(submission.score, feature_evo)
        severity_score = self._validate_severity_evolution(submission.score, severity_evo)
        
        # Calculate penalty
        penalty = self._calculate_severity_penalty(
            submission.score,
            feature_evo,
            severity_evo
        )
        
        # Combined score
        combined_score = (
            feature_score * 0.70 +
            severity_score * 0.30
        ) + penalty
        
        results.append(combined_score)
    
    return np.mean(results)
```

---

## Summary

### How Severity Evolution Relates to Penalties

| Scenario | Features | Severity T+0→T+30 | Expected Score | Penalty If Wrong |
|----------|----------|-------------------|----------------|------------------|
| Escalating risk | Expanding | [med]→[crit,high] | HIGH (0.70-1.0) | -0.15 if LOW |
| False positive | Benign | [crit]→[crit] (no new) | LOW (0.0-0.3) | -0.15 if HIGH |
| Dormant stable | Dormant | [med]→[med] | MED (0.3-0.5) | -0.10 if extreme |
| True positive | Expanding | [crit]→[crit,crit,high] | HIGH (0.8-1.0) | -0.10 if MED |

### Key Insights

1. **Severity evolution is ADDITIONAL signal**, not replacement for features
2. **Features are primary** (60-70% weight), severity is secondary (30-40% weight)
3. **Smart miners predict escalation** from T+0 features, not from T+0 severity
4. **Penalties target**: 
   - Severity copiers (just copy initial severity)
   - False positive amplifiers (high score on non-escalating critical alerts)
   - Escalation missers (low score on rapidly escalating addresses)

### Recommendation

**Implement severity evolution tracking as Phase 2 enhancement:**
- Phase 1: Current feature-based validation (works well)
- Phase 2: Add severity evolution tracking and validation
- Phase 3: Tune weights based on real miner performance

This creates a more sophisticated validation that rewards miners who can predict future risk from current features, not just copy current severity labels.