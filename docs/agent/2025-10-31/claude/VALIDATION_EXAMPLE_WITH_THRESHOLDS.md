# Validation Example: Reference Points & Decision Rules

## The Critical Question: What Makes a Prediction "Good"?

**Polski**: "Punkt odniesienia" - We need clear thresholds to determine when score 0.95 is appropriate vs when it's wrong.

**English**: Decision rules that define correct vs incorrect predictions.

---

## Part 1: Initial Alerts (Day 0 - T+0)

### SOT Provides 5 Alerts with Current Features

#### Alert A: "Clean Exchange"
```
alert_id: alert_001
address: 0xABC123...
Current Features (T+0):
  degree: 8
  total_volume_usd: $25,000
  mixer_connections: 0
  exchange_connections: 5
  days_active: 180
  avg_tx_per_day: 2.5
  velocity_score: 0.15
  anomaly_score: 0.12
```

#### Alert B: "Suspicious New Player"  
```
alert_id: alert_002
address: 0xDEF456...
Current Features (T+0):
  degree: 45
  total_volume_usd: $250,000
  mixer_connections: 8
  exchange_connections: 2
  days_active: 15
  avg_tx_per_day: 45
  velocity_score: 0.85
  anomaly_score: 0.78
```

#### Alert C: "Dormant Wallet"
```
alert_id: alert_003
address: 0xGHI789...
Current Features (T+0):
  degree: 2
  total_volume_usd: $1,500
  mixer_connections: 0
  exchange_connections: 0
  days_active: 90
  avg_tx_per_day: 0.3
  velocity_score: 0.05
  anomaly_score: 0.08
```

#### Alert D: "Moderate Activity"
```
alert_id: alert_004
address: 0xJKL012...
Current Features (T+0):
  degree: 20
  total_volume_usd: $75,000
  mixer_connections: 2
  exchange_connections: 3
  days_active: 60
  avg_tx_per_day: 12
  velocity_score: 0.45
  anomaly_score: 0.42
```

#### Alert E: "High Volume Legitimate"
```
alert_id: alert_005
address: 0xMNO345...
Current Features (T+0):
  degree: 150
  total_volume_usd: $5,000,000
  mixer_connections: 0
  exchange_connections: 50
  days_active: 365
  avg_tx_per_day: 25
  velocity_score: 0.35
  anomaly_score: 0.18
```

---

## Part 2: Miner Predictions (Day 0)

### Three Miners Analyze the Same Data

#### Miner Alpha (Conservative Model)
```
Alert A (Clean Exchange): 0.15 (LOW RISK)
Alert B (Suspicious New):  0.75 (HIGH RISK)
Alert C (Dormant):         0.10 (LOW RISK)
Alert D (Moderate):        0.45 (MEDIUM RISK)
Alert E (High Volume):     0.20 (LOW RISK)
```

#### Miner Beta (Aggressive Model)
```
Alert A (Clean Exchange): 0.05 (VERY LOW)
Alert B (Suspicious New):  0.95 (VERY HIGH RISK)
Alert C (Dormant):         0.60 (MEDIUM-HIGH RISK)
Alert D (Moderate):        0.78 (HIGH RISK)
Alert E (High Volume):     0.12 (LOW RISK)
```

#### Miner Gamma (Random/Bad Model)
```
Alert A (Clean Exchange): 0.82 (HIGH RISK) ← Wrong
Alert B (Suspicious New):  0.25 (LOW RISK) ← Wrong
Alert C (Dormant):         0.91 (VERY HIGH) ← Wrong
Alert D (Moderate):        0.33 (LOW RISK)
Alert E (High Volume):     0.67 (MEDIUM-HIGH) ← Wrong
```

---

## Part 3: Reality After 30 Days (T+30)

### What Actually Happened on the Blockchain

#### Alert A: "Clean Exchange" - STAYED CLEAN
```
address: 0xABC123...
New Features (T+30):
  degree: 9 (+1, +12.5%)
  total_volume_usd: $27,500 (+$2,500, +10%)
  mixer_connections: 0 (no change)
  exchange_connections: 6 (+1, +20%)
  days_active: 210 (+30)
  avg_tx_per_day: 2.6 (+0.1, +4%)
  velocity_score: 0.16 (+0.01)
  anomaly_score: 0.11 (-0.01)

Evolution Deltas:
  degree_delta: +1 (12.5% growth)
  volume_delta: +$2,500 (10% growth)
  mixer_delta: 0
  velocity_change: +6.7%

Pattern Classification: BENIGN_INDICATORS
  ✓ Stable degree growth (<50%)
  ✓ Steady volume growth (<30%)
  ✓ No mixer connections
  ✓ Increased exchange connections
  ✓ Low velocity change
  ✓ Decreasing anomaly score
```

#### Alert B: "Suspicious New Player" - EXPLODED
```
address: 0xDEF456...
New Features (T+30):
  degree: 180 (+135, +300%)
  total_volume_usd: $2,500,000 (+$2,250,000, +900%)
  mixer_connections: 25 (+17, +212%)
  exchange_connections: 1 (-1, -50%)
  days_active: 45 (+30)
  avg_tx_per_day: 125 (+80, +178%)
  velocity_score: 0.98 (+0.13)
  anomaly_score: 0.95 (+0.17)

Evolution Deltas:
  degree_delta: +135 (300% growth)
  volume_delta: +$2,250,000 (900% growth)
  mixer_delta: +17 (212% growth)
  velocity_change: +15.3%

Pattern Classification: EXPANDING_ILLICIT
  ✓ Massive degree growth (>200%)
  ✓ Explosive volume growth (>500%)
  ✓ Significant mixer increase (>100%)
  ✓ Decreasing exchange connections
  ✓ Very high velocity
  ✓ Rising anomaly score
```

#### Alert C: "Dormant Wallet" - STAYED DORMANT
```
address: 0xGHI789...
New Features (T+30):
  degree: 2 (no change)
  total_volume_usd: $1,650 (+$150, +10%)
  mixer_connections: 0 (no change)
  exchange_connections: 0 (no change)
  days_active: 120 (+30)
  avg_tx_per_day: 0.3 (no change)
  velocity_score: 0.05 (no change)
  anomaly_score: 0.07 (-0.01)

Evolution Deltas:
  degree_delta: 0 (0% growth)
  volume_delta: +$150 (10% growth)
  mixer_delta: 0
  velocity_change: 0%

Pattern Classification: DORMANT
  ✓ No degree growth
  ✓ Minimal volume growth (<30%)
  ✓ No mixer connections
  ✓ No new connections
  ✓ Very low velocity
  ✓ Stable/decreasing anomaly
```

#### Alert D: "Moderate Activity" - TURNED BAD
```
address: 0xJKL012...
New Features (T+30):
  degree: 85 (+65, +325%)
  total_volume_usd: $450,000 (+$375,000, +500%)
  mixer_connections: 18 (+16, +800%)
  exchange_connections: 1 (-2, -67%)
  days_active: 90 (+30)
  avg_tx_per_day: 42 (+30, +250%)
  velocity_score: 0.88 (+0.43)
  anomaly_score: 0.87 (+0.45)

Evolution Deltas:
  degree_delta: +65 (325% growth)
  volume_delta: +$375,000 (500% growth)
  mixer_delta: +16 (800% growth)
  velocity_change: +95.6%

Pattern Classification: EXPANDING_ILLICIT
  ✓ Extreme degree growth (>200%)
  ✓ Massive volume growth (>400%)
  ✓ Exploding mixer connections (>500%)
  ✓ Dropping exchange connections
  ✓ Sharp velocity increase
  ✓ Significant anomaly rise
```

#### Alert E: "High Volume Legitimate" - STAYED LEGIT
```
address: 0xMNO345...
New Features (T+30):
  degree: 165 (+15, +10%)
  total_volume_usd: $5,750,000 (+$750,000, +15%)
  mixer_connections: 0 (no change)
  exchange_connections: 55 (+5, +10%)
  days_active: 395 (+30)
  avg_tx_per_day: 27 (+2, +8%)
  velocity_score: 0.38 (+0.03)
  anomaly_score: 0.16 (-0.02)

Evolution Deltas:
  degree_delta: +15 (10% growth)
  volume_delta: +$750,000 (15% growth)
  mixer_delta: 0
  velocity_change: +8.6%

Pattern Classification: BENIGN_INDICATORS
  ✓ Modest degree growth (<30%)
  ✓ Steady volume growth (<30%)
  ✓ No mixer connections
  ✓ Growing exchange connections
  ✓ Stable velocity
  ✓ Decreasing anomaly score
```

---

## Part 4: THRESHOLDS & DECISION RULES (Punkt Odniesienia)

### Reference Point 1: Pattern Classification Thresholds

```python
PATTERN_THRESHOLDS = {
    'EXPANDING_ILLICIT': {
        'degree_growth': 200%,        # 2x or more
        'volume_growth': 300%,        # 3x or more
        'mixer_growth': 100%,         # doubled or more
        'velocity_increase': 50%,     # significant jump
        'anomaly_increase': 0.2       # +0.2 points
    },
    
    'BENIGN_INDICATORS': {
        'degree_growth': <50%,        # less than 1.5x
        'volume_growth': <30%,        # steady growth
        'mixer_growth': 0%,           # no increase
        'velocity_stable': ±20%,      # minimal change
        'anomaly_stable': ±0.1        # stable or improving
    },
    
    'DORMANT': {
        'degree_growth': <10%,        # barely any growth
        'volume_growth': <20%,        # minimal activity
        'tx_per_day': <1,             # very low activity
        'velocity': <0.2              # very low velocity
    }
}
```

### Reference Point 2: Score Interpretation Thresholds

```python
SCORE_THRESHOLDS = {
    'VERY_LOW_RISK':   0.00 - 0.20,  # Should match benign patterns
    'LOW_RISK':        0.21 - 0.40,  # Should match stable/benign
    'MEDIUM_RISK':     0.41 - 0.60,  # Uncertain, could go either way
    'HIGH_RISK':       0.61 - 0.80,  # Should match suspicious patterns
    'VERY_HIGH_RISK':  0.81 - 1.00   # Should match expanding illicit
}
```

### Reference Point 3: Validation Decision Rules

```python
VALIDATION_RULES = {
    # Cor rect Predictions (+1 point)
    'CORRECT_HIGH_RISK': {
        'condition': 'score >= 0.70 AND pattern == EXPANDING_ILLICIT',
        'reasoning': 'Predicted high risk, blockchain confirmed suspicious growth'
    },
    
    'CORRECT_LOW_RISK': {
        'condition': 'score <= 0.30 AND pattern == BENIGN_INDICATORS',
        'reasoning': 'Predicted low risk, blockchain showed stable legitimate behavior'
    },
    
    'CORRECT_DORMANT': {
        'condition': 'score <= 0.30 AND pattern == DORMANT',
        'reasoning': 'Predicted low risk, address remained inactive'
    },
    
    # Wrong Predictions (-1 point)
    'FALSE_POSITIVE': {
        'condition': 'score >= 0.70 AND pattern == BENIGN_INDICATORS',
        'reasoning': 'Predicted high risk, but blockchain showed legitimate behavior',
        'penalty': -1
    },
    
    'FALSE_NEGATIVE': {
        'condition': 'score <= 0.30 AND pattern == EXPANDING_ILLICIT',
        'reasoning': 'Predicted low risk, but blockchain showed suspicious growth',
        'penalty': -2  # More severe - missed a real threat
    },
    
    # Uncertain Cases (0 points)
    'UNCERTAIN_POSITIVE': {
        'condition': 'score >= 0.70 AND pattern == DORMANT',
        'reasoning': 'Predicted high risk, but address went dormant (inconclusive)',
        'penalty': 0
    },
    
    'UNCERTAIN_NEUTRAL': {
        'condition': '0.30 < score < 0.70',
        'reasoning': 'Neutral prediction - not counted in validation',
        'penalty': 0
    }
}
```

---

## Part 5: Validation Results for Each Miner

### Miner Alpha (Conservative Model)

#### Alert A: Clean Exchange
```
Prediction: 0.15 (LOW RISK)
Reality: BENIGN_INDICATORS
Validation Rule: CORRECT_LOW_RISK
Score: +1
Reasoning: Correctly identified legitimate exchange behavior
```

#### Alert B: Suspicious New Player
```
Prediction: 0.75 (HIGH RISK)
Reality: EXPANDING_ILLICIT
Validation Rule: CORRECT_HIGH_RISK
Score: +1
Reasoning: Correctly predicted explosive suspicious growth
```

#### Alert C: Dormant
```
Prediction: 0.10 (LOW RISK)
Reality: DORMANT
Validation Rule: CORRECT_DORMANT
Score: +1
Reasoning: Correctly identified low-activity address
```

#### Alert D: Moderate → Bad
```
Prediction: 0.45 (MEDIUM RISK)
Reality: EXPANDING_ILLICIT
Validation Rule: UNCERTAIN_NEUTRAL
Score: 0
Reasoning: Neutral prediction (0.30-0.70), not validated
```

#### Alert E: High Volume Legit
```
Prediction: 0.20 (LOW RISK)
Reality: BENIGN_INDICATORS
Validation Rule: CORRECT_LOW_RISK
Score: +1
Reasoning: Correctly identified stable legitimate business
```

**Miner Alpha Total**: 4/5 = 0.80 (80%)

---

### Miner Beta (Aggressive Model)

#### Alert A: Clean Exchange
```
Prediction: 0.05 (VERY LOW RISK)
Reality: BENIGN_INDICATORS
Validation Rule: CORRECT_LOW_RISK
Score: +1
Reasoning: Correctly identified legitimate behavior (even more confident)
```

#### Alert B: Suspicious New Player
```
Prediction: 0.95 (VERY HIGH RISK)
Reality: EXPANDING_ILLICIT
Validation Rule: CORRECT_HIGH_RISK
Score: +1
Reasoning: Correctly predicted extreme suspicious growth with high confidence
```

#### Alert C: Dormant
```
Prediction: 0.60 (MEDIUM-HIGH RISK)
Reality: DORMANT
Validation Rule: UNCERTAIN_NEUTRAL
Score: 0
Reasoning: Neutral prediction zone, address went dormant (inconclusive)
```

#### Alert D: Moderate → Bad
```
Prediction: 0.78 (HIGH RISK)
Reality: EXPANDING_ILLICIT
Validation Rule: CORRECT_HIGH_RISK
Score: +1
Reasoning: Correctly predicted it would turn suspicious!
```

#### Alert E: High Volume Legit
```
Prediction: 0.12 (LOW RISK)
Reality: BENIGN_INDICATORS
Validation Rule: CORRECT_LOW_RISK
Score: +1
Reasoning: Correctly identified legitimate business
```

**Miner Beta Total**: 4/5 = 0.80 (80%)

*Note: Beta caught Alert D that Alpha missed, but was uncertain about Alert C*

---

### Miner Gamma (Bad Model)

#### Alert A: Clean Exchange
```
Prediction: 0.82 (HIGH RISK)
Reality: BENIGN_INDICATORS
Validation Rule: FALSE_POSITIVE
Score: -1
Reasoning: Predicted high risk, but was legitimate exchange
```

#### Alert B: Suspicious New Player
```
Prediction: 0.25 (LOW RISK)
Reality: EXPANDING_ILLICIT
Validation Rule: FALSE_NEGATIVE
Score: -2
Reasoning: Predicted low risk, but exploded into major threat (SEVERE MISS)
```

#### Alert C: Dormant
```
Prediction: 0.91 (VERY HIGH RISK)
Reality: DORMANT
Validation Rule: UNCERTAIN_POSITIVE
Score: 0
Reasoning: Predicted high risk, but address went dormant (inconclusive)
```

#### Alert D: Moderate → Bad
```
Prediction: 0.33 (LOW RISK)
Reality: EXPANDING_ILLICIT
Validation Rule: FALSE_NEGATIVE
Score: -2
Reasoning: Predicted low risk, but turned suspicious (SEVERE MISS)
```

#### Alert E: High Volume Legit
```
Prediction: 0.67 (MEDIUM-HIGH RISK)
Reality: BENIGN_INDICATORS
Validation Rule: UNCERTAIN_NEUTRAL
Score: 0
Reasoning: Neutral prediction zone
```

**Miner Gamma Total**: -5/5 = -1.00 (NEGATIVE SCORE - Very Bad Model)

---

## Part 6: Final Tier 3B Scores

### Score Normalization

```python
# Raw scores (sum of validation results)
Alpha_raw = 4/5 = +0.80
Beta_raw = 4/5 = +0.80
Gamma_raw = -5/5 = -1.00

# Normalize to 0.0 - 1.0 scale
# Formula: (raw_score + max_penalty) / (max_score + max_penalty)
# Where max_penalty = 2 * num_alerts (worst case: all false negatives)
# And max_score = 1 * num_alerts (best case: all correct)

max_penalty = 2 * 5 = 10
max_score = 1 * 5 = 5
score_range = max_score + max_penalty = 15

Alpha_normalized = (4 + 10) / 15 = 14/15 = 0.933
Beta_normalized = (4 + 10) / 15 = 14/15 = 0.933
Gamma_normalized = (-5 + 10) / 15 = 5/15 = 0.333
```

### Final Tier 3B Scores

| Miner | Raw Score | Normalized Score | Interpretation |
|-------|-----------|------------------|----------------|
| Alpha | +4/5 | 0.933 (93.3%) | Excellent |
| Beta | +4/5 | 0.933 (93.3%) | Excellent |
| Gamma | -5/5 | 0.333 (33.3%) | Poor |

---

## Part 7: Why These Thresholds Work

### Threshold 1: Score >= 0.70 for "High Risk"

**Question**: Why is 0.95 considered appropriate for Alert B?

**Answer**:
```
Alert B Features at T+0:
  - Very high degree (45)
  - Very high volume ($250K)
  - Many mixer connections (8)
  - Very new account (15 days)
  - Extremely high velocity (0.85)
  - High anomaly score (0.78)

Reality at T+30:
  - Degree exploded to 180 (+300%)
  - Volume exploded to $2.5M (+900%)
  - Mixers exploded to 25 (+212%)
  - Pattern: EXPANDING_ILLICIT

Conclusion: Score 0.95 was PERFECT because:
  ✓ High confidence matched extreme suspicious features
  ✓ Prediction aligned with reality (expanding illicit)
  ✓ All indicators pointed to high risk
  ✓ Evolution confirmed the prediction
```

### Threshold 2: Score <= 0.30 for "Low Risk"

**Question**: Why is 0.15 appropriate for Alert A?

**Answer**:
```
Alert A Features at T+0:
  - Moderate degree (8)
  - Normal volume ($25K)
  - No mixer connections (0)
  - Many exchange connections (5)
  - Long active period (180 days)
  - Low velocity (0.15)
  - Very low anomaly (0.12)

Reality at T+30:
  - Slight degree growth (+12.5%)
  - Steady volume growth (+10%)
  - No mixers added
  - More exchange connections (+1)
  - Pattern: BENIGN_INDICATORS

Conclusion: Score 0.15 was PERFECT because:
  ✓ Low confidence matched legitimate indicators
  ✓ Prediction aligned with reality (benign)
  ✓ All signs pointed to normal business
  ✓ Evolution confirmed stability
```

### Threshold 3: Middle Zone (0.30 - 0.70) = Uncertain

**Question**: Why don't we validate Alert D for Miner Alpha (score 0.45)?

**Answer**:
```
Problem: Uncertain predictions are not actionable
  - Score 0.45 means "maybe risky, maybe not"
  - Could go either way
  - Can't fairly penalize OR reward

Solution: Don't count it
  - Avoids unfair penalties for honest uncertainty
  - Encourages confidence when appropriate
  - Only validates clear predictions

Exception: If miner consistently scores everything 0.40-0.60
  - Tier 2 (behavioral) will catch this
  - Distribution entropy will be very low
  - Penalized for lack of discrimination
```

---

## Part 8: Reference Points Summary

### PUNKT ODNIESIENIA (Reference Points)

#### 1. Pattern Classification Thresholds
```
EXPANDING_ILLICIT:
  - Degree growth > 200%
  - Volume growth > 300%
  - Mixer growth > 100%
  - Velocity increase > 50%
  - Anomaly increase > +0.2

BENIGN_INDICATORS:
  - Degree growth < 50%
  - Volume growth < 30%
  - No mixer increase
  - Stable velocity (±20%)
  - Stable/improving anomaly (±0.1)

DORMANT:
  - Minimal growth (<10% degree, <20% volume)
  - Very low activity (<1 tx/day)
  - Low velocity (<0.2)
```

#### 2. Score Interpretation (When to use what score)
```
0.00 - 0.20: VERY LOW RISK
  → Use for: Clear legitimate businesses, known exchanges

0.21 - 0.40: LOW RISK
  → Use for: Stable addresses, normal patterns

0.41 - 0.60: MEDIUM RISK (Uncertain)
  → Use for: Ambiguous cases, insufficient data
  → NOT counted in Tier 3B validation

0.61 - 0.80: HIGH RISK
  → Use for: Suspicious patterns, multiple red flags

0.81 - 1.00: VERY HIGH RISK
  → Use for: Severe red flags, likely illicit
```

#### 3. Validation Decision Rules
```
CORRECT (+1):
  - High score (≥0.70) + Expanding Illicit pattern
  - Low score (≤0.30) + Benign or Dormant pattern

WRONG (-1 or -2):
  - High score (≥0.70) + Benign pattern (false positive)
  - Low score (≤0.30) + Expanding Illicit (false negative, -2)

UNCERTAIN (0):
  - Medium score (0.30-0.70) + Any pattern
  - High score + Dormant pattern
```

#### 4. Normalization Formula
```
Normalized Score = (Raw Score + Max Penalty) / (Max Score + Max Penalty)

Where:
  - Raw Score = Sum of +1, -1, -2 across all alerts
  - Max Penalty = 2 × number of alerts (all false negatives)
  - Max Score = 1 × number of alerts (all correct)
```

---

## Part 9: Edge Cases & Examples

### Edge Case 1: Score 0.69 with Expanding Illicit

```
Miner Prediction: 0.69 (just below high threshold)
Reality: EXPANDING_ILLICIT
Validation: UNCERTAIN (0 points)

Reasoning: Score is in uncertain zone (0.30-0.70)
          Not counted as wrong, but also not correct
          Encourages miners to be more confident when appropriate
```

### Edge Case 2: Score 0.71 with Benign

```
Miner Prediction: 0.71 (just above high threshold)
Reality: BENIGN_INDICATORS
Validation: FALSE_POSITIVE (-1 point)

Reasoning: Crossed into "high risk" territory
          Predicted suspicious but was legitimate
          Clear false positive
```

### Edge Case 3: Score 0.50 with Any Pattern

```
Miner Prediction: 0.50 (perfectly neutral)
Reality: ANY pattern
Validation: UNCERTAIN (0 points)

Reasoning: Neutral prediction provides no useful information
          Not counted in validation
          But Tier 2 will penalize if ALL predictions are neutral
```

---

## Conclusion

### Clear Reference Points Established ✅

1. **Pattern Thresholds**: Concrete percentages for growth rates
2. **Score Zones**: Clear bands for interpretation
3. **Validation Rules**: Exact conditions for correct/wrong
4. **Normalization**: Fair scoring from -2.0 to +1.0 scale

### Example Proves the System Works

- **Good miners** (Alpha, Beta): 93.3% score - Correctly predicted most patterns
- **Bad miner** (Gamma): 33.3% score - Failed to predict patterns, made false positives/negatives

### The Answer to Your Question

**"How do we determine if score 0.95 is OK?"**

**Answer**: 
- Check what happened to the address after 30 days
- If it shows EXPANDING_ILLICIT pattern → Score 0.95 was PERFECT ✅
- If it shows BENIGN pattern → Score 0.95 was WRONG ❌
- If it went DORMANT → Score 0.95 was UNCERTAIN (no points)

**The blockchain is the "punkt odniesienia" (reference point) - what actually happened determines if the prediction was correct.**