# Anti-Gaming: Why Copying Alert Severity Fails

## The Gaming Strategy

### User's Gaming Attempt
```
"Day 0: I get critical alert, so I assign critical risk score (0.95)"
"After 30 days: Alert is still critical - I win"
```

**The Strategy**:
1. SOT provides alert with severity: "critical"
2. Miner copies severity → score: 0.95
3. 30 days later alert is still flagged
4. Miner claims victory

**Why This Seems Smart**:
- No analysis needed
- Just copy SOT's assessment
- Trivial to implement
- Should always win... right?

---

## Why This Gaming Strategy FAILS

### Reason 1: We Validate ADDRESSES, Not ALERTS

**Critical Distinction**:
- **Alert**: SOT's initial assessment (could be wrong)
- **Address**: The actual blockchain entity (objective reality)

**The Validation Flow**:

```python
# What the gamer does:
alert = get_alert('alert_12345')  
print(alert.severity)  # "critical"
miner_score = 0.95  # Copies severity

# What we validate AGAINST:
address = alert.address  # '0xABC123...'

# Get ADDRESS features (not alert assessment)
features_t0 = get_features(address, date_t0)
features_t30 = get_features(address, date_t30)

# Calculate ADDRESS evolution (objective blockchain behavior)
evolution = {
    'degree': features_t0.degree_total → features_t30.degree_total,
    'volume': features_t0.total_volume_usd → features_t30.total_volume_usd,
    # ... actual blockchain metrics
}

# Classify ADDRESS pattern
pattern = classify_pattern(evolution)
# Could be: expanding_illicit, benign_indicators, or dormant

# Validate miner score against ADDRESS pattern (not alert severity!)
if miner_score >= 0.70 and pattern == 'expanding_illicit':
    result = 'correct'
elif miner_score >= 0.70 and pattern == 'benign_indicators':
    result = 'WRONG'  # ← Gamer loses here!
```

### Scenario Where Gaming Fails

**Day 0**:
```
Alert Details:
  alert_id: alert_789
  address: 0xDEF456...
  severity: "critical"  ← SOT thinks it's critical
  description: "High volume suspicious activity"
  
Gamer's Strategy:
  score: 0.95  ← Copied from "critical" severity
```

**Day 30**:
```
Address Features Evolution:
  degree: 45 → 48 (+6.7%)  ← Barely any growth
  volume: $250K → $260K (+4%)  ← Minimal volume increase
  is_mixer_like: False (was False)  ← Not mixer behavior
  is_exchange_like: True (was True) ← Actually legitimate exchange!
  anomaly_score: 0.78 → 0.42 (-0.36)  ← Anomaly DECREASED (got more normal!)
  
Pattern Classification:
  pattern = 'benign_indicators'  ← ADDRESS turned out to be SAFE!

Validation:
  Miner score: 0.95 (HIGH RISK)
  Actual pattern: benign_indicators (SAFE)
  Result: FALSE POSITIVE (-1 point) ❌
  
Conclusion: SOT's initial "critical" alert was WRONG (false alarm)
            Gamer who copied it is also WRONG
            Loses points!
```

**Why This Happens**:
- SOT alerts are INITIAL ASSESSMENTS (can be wrong)
- SOT might flag high volume as "critical" but it could be legitimate business
- After 30 days, the ADDRESS behavior reveals truth
- Blindly copying SOT severity fails when SOT was wrong

---

## Reason 2: SOT Severity != Ground Truth

### What SOT Severity Means

From [`raw_alerts.sql`](packages/storage/schema/raw_alerts.sql):
```sql
severity String DEFAULT 'medium',  -- SOT's INITIAL assessment
suspected_address_type String DEFAULT 'unknown',
alert_confidence_score Float32,
```

**SOT Severity is**:
- ✅ Initial heuristic assessment
- ✅ Based on pattern matching
- ✅ Useful starting point

**SOT Severity is NOT**:
- ❌ Ground truth
- ❌ Guaranteed correct
- ❌ Based on future evolution
- ❌ Validated prediction

**Example of SOT Being Wrong**:
```
High volume transaction → severity: "critical"
But could be:
  - Major legitimate business (FP)
  - Exchange deposit (FP)
  - DeFi whale (FP)
  - Actual money laundering (TP)

SOT doesn't know which! That's why we need miners!
```

### Real-World Scenario

**Alert Example**:
```
SOT Alert:
  severity: "critical"
  description: "$2M rapid movement through new address"
  suspected_address_type: "mixer"
  
Reality Options:
  Option A: New exchange onboarding (legitimate)
    → After 30 days: benign_indicators pattern
    → Gamers lose (-1)
  
  Option B: Actual mixer operation (illicit)
    → After 30 days: expanding_illicit pattern
    → Gamers win (+1)
  
  Option C: Whale moving funds (neutral)
    → After 30 days: dormant pattern
    → Gamers get 0 (uncertain)

Success Rate: ~33% (random!)
```

---

## Reason 3: We Compare Against SOT Baseline

### The Real Competition

**We're NOT asking**: "Can you copy SOT?"

**We're asking**: "Can you IMPROVE on SOT?"

### Tier 2: Rank Correlation

From [`tier2_behavioral.py`](packages/validation/tier2_behavioral.py):

```python
def _check_rank_correlation(self, submissions_df, alerts_df):
    """
    Compare miner's ranking to SOT baseline ranking
    """
    # SOT baseline uses alert confidence_score
    sot_baseline_scores = alerts_df['alert_confidence_score']
    miner_scores = submissions_df['score']
    
    # Calculate rank correlation
    correlation = spearmanr(sot_baseline_scores, miner_scores)
    
    # Score: If too correlated = you're just copying SOT!
    if correlation > 0.95:
        penalty = -0.20  # Nearly identical to SOT (no value add)
    elif correlation > 0.85:
        bonus = -0.10    # Very similar to SOT (minimal value)
    elif 0.50 < correlation < 0.85:
        bonus = 0.00     # Some alignment (good)
    elif correlation < 0.30:
        penalty = -0.20  # No correlation (random)
    
    # Optimal: 0.50 to 0.85 correlation
    # Some alignment with SOT, but NOT copying
```

**Anti-Gaming Logic**:
- If miner exactly copies SOT severity → correlation ≈ 1.0 → PENALTY
- If miner slightly differs from SOT → correlation ≈ 0.6-0.8 → GOOD
- If miner completely ignores SOT → correlation ≈ 0.3 → PENALTY

**The Sweet Spot**: Use SOT as input, but add your own analysis!

---

## Reason 4: Multiple Alerts with Same Severity

### The Discrimination Problem

**SOT provides**:
```
alert_001: severity "critical", confidence 0.75
alert_002: severity "critical", confidence 0.78
alert_003: severity "critical", confidence 0.72
alert_004: severity "critical", confidence 0.76
```

**Gamer's trivial approach**:
```
All get score: 0.95  (just copy "critical")
```

**Reality after 30 days**:
```
alert_001: benign_indicators (false alarm) → Gamer WRONG (-1)
alert_002: expanding_illicit (true threat) → Gamer correct (+1)
alert_003: dormant (went inactive) → Gamer uncertain (0)
alert_004: expanding_illicit (true threat) → Gamer correct (+1)

Gamer score: (+1 -1 +0 +1) / 4 = 0.25 → Normalized: 0.625
```

**Smart miner's approach**:
```
Analyzes features for each:
alert_001: 0.45 (suspicious but uncertain) → Avoids false positive
alert_002: 0.95 (very confident) → Correct
alert_003: 0.30 (low risk) → Correct (low score for dormant)
alert_004: 0.85 (high risk) → Correct

Smart miner score: (+0 +1 +1 +1) / 4 = 0.75 → Normalized: 0.875
```

**Result**: Smart miner beats gamer!

**Why**: GameR can't discriminate WITHIN severity levels. All "critical" alerts get same score, but they have different outcomes.

---

## Reason 5: Tier 2 Catches Lazy Mining

### Distribution Entropy Check

From [`tier2_behavioral.py`](packages/validation/tier2_behavioral.py):

```python
def _check_distribution_entropy(self, submissions_df):
    """
    Check if miner's scores have good distribution
    """
    scores = submissions_df['score'].values
    
    # Calculate entropy
    hist, bins = np.histogram(scores, bins=10, range=(0, 1))
    probabilities = hist / len(scores)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    # Normalize entropy (max is log(10) ≈ 2.3)
    normalized_entropy = entropy / np.log(10)
    
    # Penalize poor distributions
    if normalized_entropy < 0.3:  # Too concentrated
        score = 0.4  # All scores clustered (lazy!)
    elif normalized_entropy > 0.7:  # Good diversity
        score = 1.0  # Well-distributed scores
    else:
        score = 0.7  # Acceptable
```

**Gaming Scenario**:
```
If gamer assigns:
  - All "critical" → 0.95
  - All "high" → 0.75
  - All "medium" → 0.50
  - All "low" → 0.25

Distribution: Very clustered (only 4 values)
Entropy: LOW (~0.4)
Tier 2 Score: 0.4 (PENALTY)
```

**Smart Miner**:
```
Assigns nuanced scores:
  [0.95, 0.87, 0.73, 0.68, 0.52, 0.41, 0.28, 0.15, 0.09, ...]

Distribution: Well-spread across range
Entropy: HIGH (~0.85)
Tier 2 Score: 1.0 (GOOD)
```

---

## The Complete Anti-Gaming Defense

### Defense Layer 1: Validate Against Address Evolution

```
NOT this:
  Alert severity at T+0 vs Alert existence at T+30 ❌

BUT this:
  Miner score at T+0 vs ADDRESS behavior T+0→T+30 ✅
```

**Why It Works**:
- Alert severity can be wrong
- Address behavior is objective
- Blockchain doesn't lie

### Defense Layer 2: Compare to SOT Baseline

```python
# Tier 2: Rank correlation
if miner_ranking == sot_baseline_ranking:
    penalty = True  # Just copying SOT!
```

**Why It Works**:
- Penalizes exact copying
- Rewards value-add analysis

### Defense Layer 3: Require Discrimination

```python
# Tier 2: Distribution entropy
if all_scores_are_same():
    penalty = True  # No discrimination!
```

**Why It Works**:
- Lazy copying gives few distinct values
- Real analysis gives nuanced scores

### Defense Layer 4: Multi-Metric Validation

```python
# Tier 3B uses THREE metrics:
- AUC: Ranking quality
- Brier: Calibration quality  
- Accuracy: Classification quality

# Can't optimize all three by copying!
```

**Why It Works**:
- AUC requires good ranking
- Brier requires calibrated probabilities
- Accuracy requires correct zones
- Copying SOT won't optimize all three

---

## Concrete Gaming Scenario Analysis

### The Gamer's Approach

**Day 0**:
```python
# Gamer's trivial code:
for alert in sot_alerts:
    if alert.severity == 'critical':
        score = 0.95
    elif alert.severity == 'high':
        score = 0.75
    elif alert.severity == 'medium':
        score = 0.50
    else:  # low
        score = 0.25
    
    submit_score(alert.alert_id, score)
```

**Day 30 Validation**:

**Tier 1: Integrity (20%)**:
```
Completeness: 1.0 ✅ (all alerts scored)
Score range: 1.0 ✅ (all in 0-1)
No duplicates: 1.0 ✅
Metadata: 1.0 ✅
Tier 1 Score: 1.0 (20% × 1.0 = 0.20)
```

**Tier 2: Behavioral (30%)**:
```
Distribution entropy: 0.4 ❌ (only 4 distinct values!)
  → Only scores: 0.95, 0.75, 0.50, 0.25
  → Very clustered distribution
  → Normalized entropy ~0.35 → Score: 0.4

Rank correlation: 0.98 ❌ (nearly identical to SOT!)
  → Spearman correlation with alert_confidence_score ≈ 0.98
  → Just copying SOT ranking
  → Penalty: -0.20 → Score: 0.3

Temporal consistency: 1.0 ✅ (deterministic copying is consistent)
  → Score: 1.0

Tier 2 Score: (0.4 + 0.3 + 1.0) / 3 = 0.57
Tier 2 Points: 30% × 0.57 = 0.17
```

**Tier 3B: Evolution (50%)**:

**Problem**: ADDRESS behavior ≠ ALERT severity

```
Example breakdown of 100 "critical" alerts:
  - 40 addresses: actually expanding_illicit → Gamer correct (+40)
  - 30 addresses: actually benign (false alarms) → Gamer wrong (-30)
  - 20 addresses: went dormant → Gamer uncertain (0)
  - 10 addresses: ambiguous → Gamer uncertain (0)

Raw Score: (+40 - 30) / 100 = +10
Normalized: (10 + 200) / 300 = 0.70

AUC: 0.55 ❌ (barely better than random)
  → Because can't discriminate WITHIN "critical" alerts
  → All get same score 0.95, but outcomes differ

Brier: 0.18 ❌ (poor calibration)
  → Many scores far from true probabilities
  → 0.95 when true is 0.0 (benign) = huge error

Accuracy: 0.40 ❌ (40% correct classification)

Tier 3B Score: 0.55×0.50 + 0.82×0.30 + 0.40×0.20 = 0.601
Tier 3B Points: 50% × 0.601 = 0.30
```

**Gamer's Final Score**:
```
Total = Tier1 + Tier2 + Tier3B
      = 0.20 + 0.17 + 0.30
      = 0.67 (67%)
```

### The Smart Miner's Approach

**Day 0**:
```python
# Smart miner's code:
for alert in sot_alerts:
    # Get FEATURES for the address
    features = get_features(alert.address)
    
    # Actual analysis
    if (features.is_mixer_like and 
        features.velocity_score > 0.8 and
        features.behavioral_anomaly_score > 0.7):
        score = 0.95
    
    elif (features.is_exchange_like and
          features.degree_total > 100 and
          features.anomaly_score < 0.3):
        score = 0.15  # Even if SOT said "critical"!
    
    # ... nuanced scoring based on features
    
    submit_score(alert.alert_id, score)
```

**Difference**:
- Uses ACTUAL FEATURES, not just severity
- Can identify false alarms
- Can discriminate within severity levels
- Adds analytical value

**Day 30 Validation**:

**Tier 2**: 
```
Distribution entropy: 0.85 ✅ (well-distributed)
Rank correlation: 0.65 ✅ (some alignment, not copying)
Tier 2 Score: 0.85
Tier 2 Points: 30% × 0.85 = 0.26
```

**Tier 3B**:
```
AUC: 0.88 ✅ (excellent discrimination)
Brier: 0.04 → 0.96 ✅ (excellent calibration)
Accuracy: 0.85 ✅ (85% correct)

Tier 3B Score: 0.88×0.50 + 0.96×0.30 + 0.85×0.20 = 0.898
Tier 3B Points: 50% × 0.898 = 0.45
```

**Smart Miner's Final Score**:
```
Total = 0.20 + 0.26 + 0.45 = 0.91 (91%)
```

**Winner**: Smart Miner (0.91) beats Gamer (0.67) by 24 points!

---

## Why SOT Alerts Can Be Wrong

### Source of SOT Alerts

SOT alerts are generated by:
1. **Pattern detection algorithms** (can have false positives)
2. **Statistical thresholds** (catch volume spikes, unusual patterns)
3. **Heuristic rules** (rapid movements, new addresses)
4. **Machine learning models** (can misclassify)

**Not generated by**:
- ❌ Future knowledge
- ❌ Perfect ground truth
- ❌ Human verification

### False Positive Examples

**Scenario 1**: New Exchange Launch
```
Day 0:
  - High volume ($5M)
  - New address (30 days old)
  - Many connections (200+)
  - SOT flags as: severity "critical"

Day 30:  
  - Stable exchange patterns
  - Growing legitimate connections
  - Pattern: benign_indicators

Result: SOT alert was FALSE ALARM
        Gamer who copied loses points
```

**Scenario 2**: Whale Selling NFTs
```
Day 0:
  - Massive volume spike ($10M in 1 day)
  - Unusual pattern
  - SOT flags as: severity "critical"

Day 30:
  - Whale finished selling
  - Address went dormant
  - Pattern: dormant

Result: SOT alert was TEMPORARY spike
        Gamer who copied gets 0 points (uncertain)
```

**Scenario 3**: DeFi Protocol Interaction
```
Day 0:
  - Complex transaction patterns
  - High frequency
  - SOT flags as: severity "high" (suspected mixer)

Day 30:
  - Continued DeFi activity
  - Stable legitimate patterns
  - Pattern: benign_indicators

Result: SOT misclassified DeFi as mixer
        Gamer who copied loses points
```

---

## The Correct Strategy: Feature Analysis

### What Smart Miners Do

**Instead of**:
```python
score = severity_to_score(alert.severity)  # Lazy gaming
```

**Do this**:
```python
# Get ACTUAL address features
features = get_features(alert.address)

# Analyze multiple dimensions
risk_factors = []

# Network position
if features.degree_total > 100 and features.pagerank > 0.001:
    risk_factors.append(('hub_risk', 0.15))

# Mixer behavior
if features.is_mixer_like:
    risk_factors.append(('mixer', 0.25))
elif features.is_exchange_like:
    risk_factors.append(('exchange', -0.30))  # Reduces risk

# Volume patterns
if features.total_volume_usd > 1000000 and features.velocity_score > 0.8:
    risk_factors.append(('high_velocity', 0.20))

# Anomaly scores
if features.behavioral_anomaly_score > 0.7:
    risk_factors.append(('anomaly', 0.15))

# Age consideration
if features.account_age_days < 30 and features.degree_total > 50:
    risk_factors.append(('new_active', 0.20))

# Calculate final score from factors
base = 0.50
for factor, adjustment in risk_factors:
    base += adjustment

score = max(0.0, min(1.0, base))
```

**This approach**:
- Uses ACTUAL features (not just severity)
- Considers multiple dimensions
- Can identify false alarms
- Discriminates within severity levels
- Adds real analytical value

---

## Summary: Why Gaming Fails

### The Gaming Strategy
```
"Copy SOT severity → Convert to score → Wait 30 days → Win"
```

### Why It Fails

1. **Validates Against Address, Not Alert** ❌
   - SOT alert severity can be wrong
   - Address behavior is ground truth
   - Many "critical" alerts are false alarms

2. **SOT Baseline Comparison** ❌
   - Tier 2 penalizes high correlation with SOT
   - Must ADD VALUE beyond SOT

3. **No Discrimination** ❌
   - All "critical" get same score
   - But outcomes differ (expanding vs benign vs dormant)
   - AUC suffers from lack of discrimination

4. **Poor Distribution** ❌
   - Only 4 distinct values
   - Low entropy
   - Tier 2 penalty

5. **Bad Calibration** ❌
   - 0.95 for all "critical" including false alarms
   - High Brier score (poor calibration)
   - Tier 3B penalty

### Expected Performance

**Gamer Score**: ~0.65-0.70 (65-70%)
- Tier 1: 1.0 (passes basic checks)
- Tier 2: 0.55 (poor distribution, high SOT correlation)
- Tier 3B: 0.60 (poor discrimination and calibration)

**Smart Miner Score**: ~0.85-0.92 (85-92%)
- Tier 1: 1.0 (passes basic checks)
- Tier 2: 0.85 (good distribution, moderate SOT correlation)
- Tier 3B: 0.90 (excellent discrimination and calibration)

**Winner**: Smart miners beat gamers by ~20-25 points!

---

## Conclusion

### Your Gaming Strategy
```
"Day 0: Get critical alert → assign critical score (0.95)"
"Day 30: Alert still critical → I win"
```

### Why It Doesn't Work

**The flaw in the assumption**: "Alert still critical"

**The reality**:
- We don't check if ALERT is still critical
- We check if ADDRESS showed expanding_illicit BEHAVIOR
- Many "critical" alerts are false alarms
- Addresses evolve independently of initial alert assessment

**Example**:
```
Day 0:
  Alert severity: "critical" (SOT's guess)
  Gamer score: 0.95 (copied)

Day 30:
  Address behavior: benign_indicators (blockchain reality)
  Gamer result: FALSE POSITIVE (-1)

The alert might still exist in SOT's database as "critical",
but the ADDRESS didn't behave suspiciously!
```

### The Lesson

**Gaming by copying SOT severity fails because**:
1. We validate against ADDRESS behavior, not ALERT persistence
2. SOT alerts are imperfect (contain false positives)
3. Tier 2 penalizes high correlation with SOT
4. Need discrimination within severity levels
5. Need calibrated probabilities, not binary copying

**To succeed, miners must**:
1. Analyze ACTUAL address features
2. Build discriminative models
3. Calibrate confidence properly
4. Add value beyond SOT baseline
5. Handle false alarms correctly

The validation system is designed to reward genuine analysis, not lazy copying!