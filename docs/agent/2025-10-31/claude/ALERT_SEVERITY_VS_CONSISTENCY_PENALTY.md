# Alert Severity vs Consistency Penalty - Detailed Analysis

## Critical Question

**"What if given address has critical, high, medium, low risk alerts? How does it relate to penalties?"**

---

## Understanding the Distinction

### Alert Severity (Feature/Input)
```sql
-- From raw_alerts table
severity String DEFAULT 'medium',  -- critical/high/medium/low
typology_type String,              -- mixing/layering/structuring/etc
```

This is **what the validator detected** - a feature, not ground truth.

### Miner Score (Prediction/Output)
```json
{
  "alert_id": "alert_001",
  "score": 0.95  // Miner's prediction (0.0 to 1.0)
}
```

This is **miner's assessment** of true risk probability.

### Address Evolution (Ground Truth)
```sql
-- From feature_evolution_tracking
degree_delta Int32,           -- +250% growth
volume_delta Decimal128(18),  -- +500% growth
pattern_classification String -- expanding/benign/dormant
```

This is **what actually happened** to the address behavior.

---

## Key Insight: We Validate Against ADDRESS, Not ALERT

**Wrong Thinking:**
```
alert_001.severity = "critical" → score should be 0.95
alert_002.severity = "low" → score should be 0.15
Different severities = inconsistent scoring = penalty ✗
```

**Correct Thinking:**
```
Address 0xABC evolved: degree +250%, volume +500%, mixer_like=true
ALL alerts for 0xABC should reflect this SAME address behavior
Scores should align with ADDRESS evolution, not alert severity ✓
```

---

## Real-World Scenario

### Scenario: Multi-Pattern Address

```
Address: 0xABC (known money launderer)
Timeline:
├─ T+0: Alert generated
├─ T+7: Degree +50%, volume +100%
├─ T+14: Degree +150%, volume +250%
├─ T+21: Degree +250%, volume +400%
└─ T+30: Degree +300%, volume +500% + contacted mixer

Alerts detected at T+0:
1. alert_001: Mixing pattern (severity=critical)
2. alert_002: Layering pattern (severity=high)
3. alert_003: Structuring (severity=medium)
4. alert_004: Unusual timing (severity=low)
```

### Miner Submission Options

#### Option A: Smart Miner (CORRECT) ✓

**Reasoning:** "Same address, same evolution data → all patterns are real"

```python
Scores based on address evolution (expanding pattern):
- alert_001 (mixing): 0.92      # High confidence
- alert_002 (layering): 0.88    # High confidence
- alert_003 (structuring): 0.85 # High confidence
- alert_004 (timing): 0.80      # Still high (same address!)

Validation:
- Address avg: (0.92 + 0.88 + 0.85 + 0.80) / 4 = 0.8625
- Std dev: 0.05 (LOW variance)
- Consistency penalty: 0.0
- Evolution match: GOOD (expanding pattern confirmed)
- Final score: 0.86 ✓✓✓
```

**Why no penalty?** Variance is LOW (0.05), and all scores align with the SAME address evolution (expanding).

#### Option B: Naive Miner (WRONG) ✗

**Reasoning:** "Just copy alert severity"

```python
Scores based on alert severity (ignores address evolution):
- alert_001 (critical): 0.95
- alert_002 (high): 0.70
- alert_003 (medium): 0.50
- alert_004 (low): 0.20

Validation:
- Address avg: (0.95 + 0.70 + 0.50 + 0.20) / 4 = 0.5875
- Std dev: 0.31 (HIGH variance! 0.31 > 0.15 threshold)
- Consistency penalty: -0.1
- Evolution match: POOR (should all be HIGH for expanding)
- Final score: 0.58 - 0.1 = 0.48 ✗✗✗
```

**Why penalty?** High variance (0.31) suggests miner is NOT using address evolution, just copying severity.

#### Option C: Gaming Miner (WRONG) ✗✗

**Reasoning:** "Random scores to confuse system"

```python
Random scores:
- alert_001 (mixing): 0.90
- alert_002 (layering): 0.15  # Drastically different!
- alert_003 (structuring): 0.85
- alert_004 (timing): 0.25

Validation:
- Address avg: (0.90 + 0.15 + 0.85 + 0.25) / 4 = 0.5375
- Std dev: 0.36 (VERY HIGH variance! 0.36 >> 0.15)
- Consistency penalty: -0.15 (double penalty)
- Evolution match: TERRIBLE (contradictory)
- Final score: 0.54 - 0.15 = 0.39 ✗✗✗
```

**Why severe penalty?** Extreme variance (0.36) indicates no coherent model - just random guessing.

---

## The Consistency Rule

### What Gets Penalized?

**HIGH variance in scores for SAME address = Penalty**

```python
def calculate_consistency_penalty(scores_for_same_address):
    std_dev = np.std(scores)
    
    if std_dev < 0.10:
        return 0.0      # Consistent scoring
    elif std_dev < 0.15:
        return -0.05    # Minor inconsistency
    elif std_dev < 0.25:
        return -0.10    # Moderate inconsistency
    else:
        return -0.15    # Severe inconsistency (gaming?)
```

### What Does NOT Get Penalized?

**Different scores for different addresses**

```python
# This is GOOD - different addresses can have different risks:
Address 0xAAA: [0.90, 0.85, 0.88]  # avg=0.88, std=0.02 → no penalty
Address 0xBBB: [0.15, 0.20, 0.18]  # avg=0.18, std=0.02 → no penalty
Address 0xCCC: [0.95]               # avg=0.95, std=0.0  → no penalty

Final: (0.88 + 0.18 + 0.95) / 3 = 0.67
```

**Moderate variance IF justified by different typologies**

```python
# This might be OK if well-calibrated:
Address 0xDDD with multiple patterns:
- Mixing (confirmed in evolution): 0.92
- Layering (partial evidence): 0.65
- Timing (weak signal): 0.40

Std = 0.26 (high but explainable)

# We check: Do these scores correlate with feature evidence?
if mixing_evidence_high and score_high: ✓
if layering_evidence_medium and score_medium: ✓
if timing_evidence_low and score_low: ✓

# Calibration check - if well-calibrated, reduce penalty
penalty = -0.10 → reduced to -0.05
```

---

## Validation Logic with Alert Context

### Enhanced Consistency Check

```python
def validate_address_consistency(submissions, address, evolution):
    """Check if scores for same address make sense"""
    
    alerts = filter_by_address(submissions, address)
    scores = [a.score for a in alerts]
    
    # 1. Basic variance check
    std_dev = np.std(scores)
    base_penalty = calculate_base_penalty(std_dev)
    
    # 2. Evolution alignment check
    expected_range = get_expected_range_from_evolution(evolution)
    # expanding → [0.70, 1.00]
    # benign → [0.00, 0.30]
    # dormant → [0.15, 0.25]
    
    in_range_count = sum(1 for s in scores if expected_range[0] <= s <= expected_range[1])
    alignment_ratio = in_range_count / len(scores)
    
    # 3. Pattern-specific adjustment
    typologies = [get_typology(a.alert_id) for a in alerts]
    if has_high_confidence_patterns(typologies, evolution):
        # Multiple strong patterns confirmed → scores should be similar
        if std_dev > 0.20:
            base_penalty *= 1.5  # Increase penalty
    else:
        # Mixed confidence → some variance expected
        if std_dev < 0.30:
            base_penalty *= 0.7  # Reduce penalty
    
    # 4. Final consistency score
    consistency_score = 1.0 - std_dev - base_penalty
    
    return {
        'penalty': base_penalty,
        'consistency_score': consistency_score,
        'alignment_ratio': alignment_ratio,
        'variance': std_dev
    }
```

### Pattern-Specific Expectations

```python
def get_expected_score_range(evolution, typology):
    """Expected score range based on evolution + typology"""
    
    pattern = evolution.pattern_classification
    
    if pattern == "expanding":
        # High-growth illicit behavior
        if typology in ["mixing", "layering"]:
            return (0.85, 1.00)  # Should be very high
        elif typology in ["structuring"]:
            return (0.70, 0.90)  # High but varied
        else:
            return (0.60, 0.85)  # Moderate-high
    
    elif pattern == "benign":
        # Normal behavior
        if typology in ["mixing", "layering"]:
            return (0.00, 0.20)  # False positive
        else:
            return (0.00, 0.35)  # Low risk
    
    elif pattern == "dormant":
        # Inactive pattern
        return (0.10, 0.30)  # Low-moderate
    
    return (0.30, 0.70)  # Default: uncertain
```

---

## Concrete Examples

### Example 1: Legitimate Variance (No Penalty)

```
Address 0xMIXER (confirmed mixer):
- Evolution: degree +400%, volume +800%, mixer_like=true

Alerts:
1. Mixing detection (high confidence ML) → score 0.95 ✓
2. Layering (medium confidence) → score 0.85 ✓
3. Unusual timing (low confidence) → score 0.75 ✓

Std = 0.10 (LOW)
Expected ranges: [0.85-1.0], [0.70-0.90], [0.60-0.85]
All scores in expected range ✓
Penalty: 0.0
```

**Interpretation:** Variance reflects different typology confidence levels, NOT inconsistent modeling. This is GOOD scoring.

### Example 2: Unjustified Variance (Penalty)

```
Address 0xMIXER (same as above):

Alerts:
1. Mixing detection → score 0.95 ✓
2. Layering → score 0.30 ✗ (too low for expanding pattern!)
3. Unusual timing → score 0.80 ✓

Std = 0.33 (HIGH)
Expected ranges: [0.85-1.0], [0.70-0.90], [0.60-0.85]
Score 2 OUT of expected range ✗
Penalty: -0.15
```

**Interpretation:** Layering score (0.30) contradicts address evolution (expanding). Suggests miner isn't using feature evolution consistently.

### Example 3: Different Severities, Same High Scores (No Penalty)

```
Address 0xLAUNDER (sophisticated launderer):
- Evolution: degree +350%, volume +600%, all indicators HIGH

Alerts:
1. Mixing (severity=critical) → score 0.92 ✓
2. Layering (severity=high) → score 0.90 ✓
3. Structuring (severity=medium) → score 0.88 ✓
4. Timing (severity=low) → score 0.85 ✓

Std = 0.03 (VERY LOW)
All scores HIGH despite varying severities
All in expected range [0.85-1.0] ✓
Penalty: 0.0
```

**Interpretation:** Miner correctly identified that ALL patterns from this address are HIGH risk, regardless of individual alert severity. This is EXCELLENT scoring!

---

## Summary

### Penalty Triggers

❌ **NOT penalized:**
- Different scores for different alert severities (if justified by evolution)
- Different scores for different addresses
- Moderate variance (std < 0.15)
- Variance that correlates with typology confidence

✅ **GETS penalized:**
- High variance (std > 0.15) for same address
- Scores contradicting address evolution
- Random/inconsistent scoring patterns
- Extreme variance (std > 0.25)

### The Core Principle

**"All alerts from the same address should reflect the SAME address evolution, regardless of individual alert severity"**

```python
# Good Miner Logic:
address_risk = analyze_evolution(address)  # 0.90 (expanding)
for alert in address_alerts:
    base_score = address_risk  # Start with address assessment
    typology_confidence = get_confidence(alert.typology)
    final_score = base_score * typology_confidence
    # Results in similar scores (0.85-0.95) with low variance

# Bad Miner Logic:
for alert in address_alerts:
    score = copy_severity(alert.severity)  # Ignores evolution!
    # Results in high variance (0.20-0.95)
```

### Implementation Recommendation

Add typology-aware consistency checking:

```python
def enhanced_consistency_check(submissions_per_address):
    penalty = base_consistency_penalty(std_dev)
    
    # Adjust for pattern context
    if all_strong_patterns and high_variance:
        penalty *= 1.5  # Increase - should be consistent
    
    if mixed_confidence_patterns and moderate_variance:
        penalty *= 0.7  # Decrease - some variance expected
    
    return penalty
```

This balances:
- Detecting gaming (random scoring)
- Allowing legitimate variance (different typology confidences)
- Rewarding address-centric modeling (evolution-based)