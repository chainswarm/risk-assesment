# The Missing Link: How Risk Scores Correlate with Features

## The Critical Problem You Identified

**User's Question**: "There is no correlation between alert/cluster risk score AND features! How will you know how risk score should be scored in context of future features?"

**The Problem**:
```
Miner submits:          score = 0.95
Features evolve:        degree +300%, volume +900%
Question:               Should the score be 0.95? 0.85? 0.70?
Problem:                We have no reference for what score SHOULD be!
```

**You're absolutely right** - we need to establish the correlation between feature patterns and expected risk scores.

---

## Solution 1: AUC - Ranking Quality (NOT Absolute Scores)

### The Key Insight

**We DON'T need to know the exact score value!**

What we need to know:
- ✅ Do HIGH scores correlate with BAD patterns?
- ✅ Do LOW scores correlate with GOOD patterns?
- ✅ Can the miner DISCRIMINATE between risky and safe?

This is what **AUC (Area Under ROC Curve)** measures!

### How AUC Works with Feature Evolution

**Step 1: Classify All Alerts by Pattern**

```python
# After 30 days, classify each alert:
alerts = [
    {'alert_id': 'alert_001', 'pattern': 'benign_indicators'},      # GOOD
    {'alert_id': 'alert_002', 'pattern': 'expanding_illicit'},      # BAD
    {'alert_id': 'alert_003', 'pattern': 'dormant'},                # GOOD
    {'alert_id': 'alert_004', 'pattern': 'expanding_illicit'},      # BAD
    {'alert_id': 'alert_005', 'pattern': 'benign_indicators'},      # GOOD
    {'alert_id': 'alert_006', 'pattern': 'expanding_illicit'},      # BAD
]

# Convert patterns to binary labels:
labels = [0, 1, 0, 1, 0, 1]  # 1 = expanding_illicit, 0 = benign/dormant
```

**Step 2: Get Miner's Scores**

```python
# Retrieve miner's predictions from 30 days ago:
scores = [
    0.15,  # alert_001 (predicted low risk)
    0.95,  # alert_002 (predicted high risk)
    0.10,  # alert_003 (predicted low risk)
    0.85,  # alert_004 (predicted high risk)
    0.25,  # alert_005 (predicted low risk)
    0.78,  # alert_006 (predicted high risk)
]
```

**Step 3: Calculate AUC**

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(
    y_true=labels,   # [0, 1, 0, 1, 0, 1]
    y_score=scores   # [0.15, 0.95, 0.10, 0.85, 0.25, 0.78]
)
# Result: AUC = 1.0 (perfect discrimination!)
```

### Why This Works

**AUC measures RANKING**, not absolute values:
- It doesn't care if scores are 0.95 or 0.75
- It cares if HIGH scores go to expanding_illicit patterns
- It cares if LOW scores go to benign patterns

**Perfect AUC (1.0)**: All expanding_illicit alerts ranked higher than all benign alerts
**Random AUC (0.5)**: No discrimination ability
**Bad AUC (<0.5)**: Backwards - high scores on safe addresses!

### Example Comparison

**Miner A** (Correct discrimination):
```
Scores:  [0.95, 0.85, 0.78] for expanding_illicit
Scores:  [0.25, 0.15, 0.10] for benign
AUC: 1.0 ✅
```

**Miner B** (Wrong discrimination):
```
Scores:  [0.25, 0.15, 0.10] for expanding_illicit ← LOW scores on BAD patterns!
Scores:  [0.95, 0.85, 0.78] for benign ← HIGH scores on GOOD patterns!
AUC: 0.0 ❌ (completely backwards!)
```

**Miner C** (No discrimination):
```
Scores:  [0.50, 0.51, 0.49] for expanding_illicit
Scores:  [0.48, 0.52, 0.50] for benign
AUC: 0.5 (random guessing)
```

---

## Solution 2: Reference Model (Ideal Scores)

### Building a Reference Scorer

**Concept**: Create a "baseline" scorer that assigns ideal scores based on feature evolution:

```python
def calculate_ideal_score(pattern, feature_deltas):
    """
    What SHOULD the score be for this pattern?
    """
    
    if pattern == 'expanding_illicit':
        # Calculate intensity of suspicious behavior
        degree_intensity = min(feature_deltas['degree_growth_pct'] / 3.0, 1.0)  # /3 = normalize 300% → 1.0
        volume_intensity = min(feature_deltas['volume_growth_pct'] / 5.0, 1.0)  # /5 = normalize 500% → 1.0
        
        # Average intensities
        base_score = (degree_intensity + volume_intensity) / 2
        
        # Boost for additional red flags
        if feature_deltas['anomaly_delta'] > 0.15:
            base_score += 0.10
        if features_t30.is_mixer_like:
            base_score += 0.10
        
        # Ideal score for expanding_illicit: 0.70 to 1.00
        ideal_score = max(0.70, min(1.00, base_score))
    
    elif pattern == 'benign_indicators':
        # Ideal score for benign: 0.00 to 0.30
        ideal_score = 0.15
    
    elif pattern == 'dormant':
        # Ideal score for dormant: 0.00 to 0.30
        ideal_score = 0.10
    
    else:  # ambiguous
        ideal_score = 0.50
    
    return ideal_score
```

### Compare Miner Score to Ideal Score

```python
# Example: Alert B after 30 days
pattern = 'expanding_illicit'
feature_deltas = {
    'degree_growth_pct': 3.0,   # 300%
    'volume_growth_pct': 9.0,   # 900%
    'anomaly_delta': 0.17
}
features_t30 = {'is_mixer_like': True}

# Calculate ideal score
ideal_score = calculate_ideal_score(pattern, feature_deltas)
# Result: ideal_score = 0.95
# Calculation: (1.0 + 1.0)/2 = 1.0, +0.10 (anomaly), +0.10 (mixer) = 1.20 → capped at 1.00

# Get miner's score
miner_score = 0.95

# Calculate error
error = abs(miner_score - ideal_score)
# Result: abs(0.95 - 1.00) = 0.05 (very small error!)

# Score based on error
if error < 0.10:
    result = 'excellent'  # Within 0.10 of ideal
elif error < 0.20:
    result = 'good'       # Within 0.20 of ideal
else:
    result = 'poor'       # More than 0.20 off
```

---

## Solution 3: Hybrid Approach (RECOMMENDED)

### Combine Both Methods

**Method 1 (Primary)**: AUC for ranking quality
- Measures discrimination ability
- Doesn't require exact scores
- Industry standard
- **Weight: 70%**

**Method 2 (Secondary)**: Brier Score for calibration
- Measures how well-calibrated probabilities are
- Penalizes confident wrong predictions
- Rewards accurate confidence
- **Weight: 30%**

### How Brier Score Adds Calibration

**Brier Score Formula**:
```python
brier_score = mean((predicted_score - actual_outcome)²)
```

**For Feature Evolution**:
```python
# Convert pattern to probability
def pattern_to_probability(pattern):
    if pattern == 'expanding_illicit':
        return 1.0  # Definitely suspicious
    elif pattern == 'benign_indicators':
        return 0.0  # Definitely safe
    elif pattern == 'dormant':
        return 0.2  # Probably safe (small chance of reactivation)
    else:  # ambiguous
        return 0.5  # Uncertain

# Calculate Brier score
brier = mean([
    (0.95 - 1.0)²,  # Alert B: predicted 0.95, actual 1.0 (expanding) = 0.0025
    (0.15 - 0.0)²,  # Alert A: predicted 0.15, actual 0.0 (benign) = 0.0225
    (0.10 - 0.2)²,  # Alert C: predicted 0.10, actual 0.2 (dormant) = 0.01
])
# Result: average of squared errors

# Lower Brier = better calibration
# Brier = 0.0 → perfect calibration
# Brier = 0.25 → random/bad calibration
```

**Why Brier Matters**?:
- Penalizes over-confidence: Predicting 0.99 when should be 0.70
- Penalizes under-confidence: Predicting 0.60 when should be 0.95
- Rewards calibrated confidence

---

## The Complete Validation Formula

### Tier 3B Evolution Score

```python
# Step 1: Classify patterns (from feature evolution)
patterns = classify_all_patterns(features_t0, features_t30)
# Results: [benign, expanding_illicit, dormant, expanding_illicit, benign]

# Step 2: Convert to binary labels
labels = [
    1 if p == 'expanding_illicit' else 0 
    for p in patterns
]
# Results: [0, 1, 0, 1, 0]

# Step 3: Get miner scores
scores = get_miner_scores(miner_id, processing_date)
# Results: [0.15, 0.95, 0.10, 0.85, 0.25]

# Step 4: Calculate AUC (ranking quality)
auc_score = roc_auc_score(labels, scores)
# Result: 1.0 (perfect ranking)

# Step 5: Convert patterns to probabilities
probabilities = [pattern_to_probability(p) for p in patterns]
# Results: [0.0, 1.0, 0.2, 1.0, 0.0]

# Step 6: Calculate Brier (calibration quality)
brier_score = mean([(s - p)² for s, p in zip(scores, probabilities)])
# Result: small value (well-calibrated)

# Step 7: Combine metrics
evolution_auc = auc_score                    # 1.0
evolution_brier = 1.0 - brier_score         # Invert so higher is better
evolution_pattern_accuracy = calculate_pattern_accuracy(scores, patterns)  # % correct classifications

# Step 8: Final Tier 3B score
tier3b_score = (
    evolution_auc * 0.50 +                   # Ranking quality
    evolution_brier * 0.30 +                 # Calibration quality
    evolution_pattern_accuracy * 0.20        # Classification accuracy
)
```

---

## The Reference Points (Punkt Odniesienia) - COMPLETE

### Reference 1: Pattern → Probability Mapping

```python
PATTERN_TO_PROBABILITY = {
    'expanding_illicit': 1.0,      # Definitely suspicious (100%)
    'benign_indicators': 0.0,      # Definitely safe (0%)
    'dormant': 0.2,                # Probably safe (20% risk)
    'ambiguous': 0.5               # Uncertain (50%)
}
```

**This is the "ground truth probability" based on observed behavior!**

### Reference 2: Expected Score Ranges per Pattern

```python
EXPECTED_SCORE_RANGES = {
    'expanding_illicit': (0.70, 1.00),   # Should predict high risk
    'benign_indicators': (0.00, 0.30),   # Should predict low risk
    'dormant': (0.00, 0.30),             # Should predict low risk
    'ambiguous': (0.30, 0.70)            # Uncertain is OK
}
```

### Reference 3: Classification Rules

```python
def classify_prediction(score, pattern):
    """
    Determine if score is appropriate for pattern
    """
    expected_min, expected_max = EXPECTED_SCORE_RANGES[pattern]
    
    if expected_min <= score <= expected_max:
        return 'correct_classification'  # Score in expected range
    elif score > expected_max:
        return 'over_confident'  # Score too high
    elif score < expected_min:
        return 'under_confident'  # Score too low
```

---

## The Complete Picture: Three-Layer Validation

### Layer 1: Classification Correctness (Binary)

**Question**: Did miner put score in right zone?

```python
score = 0.95
pattern = 'expanding_illicit'
expected_range = (0.70, 1.00)

if 0.70 <= 0.95 <= 1.00:
    classification = 'CORRECT'  # ✅ Score in right zone
```

### Layer 2: Ranking Quality (AUC)

**Question**: Are higher scores for worse patterns?

```python
scores = [0.15, 0.95, 0.10, 0.85, 0.25]
labels = [0, 1, 0, 1, 0]  # 1 = expanding_illicit, 0 = benign/dormant

auc = roc_auc_score(labels, scores)
# Result: 1.0 ✅ Perfect ranking
```

**What AUC Tells Us**:
- All expanding_illicit alerts scored higher than all benign alerts
- Order is perfect, exact values don't matter
- 0.95 and 0.85 both > 0.25, 0.15, 0.10 → good!

### Layer 3: Calibration Quality (Brier)

**Question**: Are scores well-calibrated to probabilities?

```python
scores = [0.15, 0.95, 0.10, 0.85, 0.25]
true_probs = [0.0, 1.0, 0.2, 1.0, 0.0]  # Based on patterns

brier = mean([
    (0.15 - 0.0)²,   # 0.0225
    (0.95 - 1.0)²,   # 0.0025
    (0.10 - 0.2)²,   # 0.01
    (0.85 - 1.0)²,   # 0.0225
    (0.25 - 0.0)²,   # 0.0625
])
# Result: 0.024 (excellent calibration!)
```

**What Brier Tells Us**:
- Score 0.95 is close to true probability 1.0 ✅
- Score 0.15 is close to true probability 0.0 ✅
- Small errors = well-calibrated

---

## How We Establish "True Probability" from Features

### The Feature → Probability Function

```python
def calculate_true_probability_from_features(features_t0, features_t30):
    """
    Based on feature evolution, what's the TRUE risk probability?
    """
    
    # Calculate evolution metrics
    degree_growth = (features_t30.degree_total - features_t0.degree_total) / features_t0.degree_total
    volume_growth = (features_t30.total_volume_usd - features_t0.total_volume_usd) / features_t0.total_volume_usd
    
    anomaly_increase = features_t30.behavioral_anomaly_score - features_t0.behavioral_anomaly_score
    velocity_increase = features_t30.velocity_score - features_t0.velocity_score
    
    # Start with base probability
    probability = 0.5
    
    # Adjust based on ACTUAL feature evolution
    
    # Degree growth contribution
    if degree_growth > 3.0:        # >300%
        probability += 0.30
    elif degree_growth > 2.0:      # >200%
        probability += 0.20
    elif degree_growth < 0.1:      # <10%
        probability -= 0.20
    
    # Volume growth contribution
    if volume_growth > 5.0:        # >500%
        probability += 0.30
    elif volume_growth > 3.0:      # >300%
        probability += 0.20
    elif volume_growth < 0.2:      # <20%
        probability -= 0.20
    
    # Mixer behavior
    if features_t30.is_mixer_like:
        probability += 0.15
    
    # Exchange behavior (inverse)
    if features_t30.is_exchange_like:
        probability -= 0.20
    
    # Anomaly increase
    if anomaly_increase > 0.20:
        probability += 0.15
    elif anomaly_increase > 0.10:
        probability += 0.10
    
    # Velocity increase
    if velocity_increase > 0.15:
        probability += 0.10
    
    # Pagerank growth (network importance)
    pagerank_growth = (features_t30.pagerank - features_t0.pagerank) / features_t0.pagerank
    if pagerank_growth > 5.0:      # >500%
        probability += 0.10
    
    # Bound to [0, 1]
    probability = max(0.0, min(1.0, probability))
    
    return probability
```

### Example Calculation

**Alert B** (expanding_illicit):
```python
features_t0 = {'degree_total': 45, 'total_volume_usd': 250000, ...}
features_t30 = {'degree_total': 180, 'total_volume_usd': 2500000, ...}

# Calculate true probability
degree_growth = 3.0     # 300%
volume_growth = 9.0     # 900%
anomaly_increase = 0.17
velocity_increase = 0.13
is_mixer_like = True
pagerank_growth = 5.33  # 533%

probability = 0.5               # base
probability += 0.30             # degree > 300%
probability += 0.30             # volume > 500%
probability += 0.15             # is_mixer_like
probability += 0.15             # anomaly > 0.15
probability += 0.10             # velocity > 0.15
probability += 0.10             # pagerank > 500%
probability = min(1.0, 1.60)    # = 1.0 (capped)

# True probability = 1.0 (definitely suspicious)
```

**Miner's Score**: 0.95

**Evaluation**:
```python
# Error
error = abs(0.95 - 1.0) = 0.05

# Squared error (Brier)
squared_error = (0.95 - 1.0)² = 0.0025

# Conclusion: Excellent! Score very close to true probability
```

---

## The Answer to Your Question

### "How will you know how risk score should be scored?"

**Complete Answer**:

#### Method 1: AUC (Ranking)
We DON'T need exact score values. We just need:
- HIGH scores → expanding_illicit patterns
- LOW scores → benign/dormant patterns
- Measured by AUC (area under ROC curve)

#### Method 2: Feature-Based True Probability
We CAN calculate ideal scores:
```
True Probability = f(feature_evolution)

Where f() considers:
  - Degree growth rate
  - Volume growth rate
  - Mixer behavior changes
  - Anomaly score changes
  - Velocity changes
  - Pagerank changes
  - All other ACTUAL feature deltas
```

#### Method 3: Brier Score (Calibration)
```
Brier = mean((miner_score - true_probability)²)

Lower Brier = better calibration
```

### Combined Tier 3B Score

```python
tier3b_score = (
    evolution_auc * 0.50 +           # Can you rank correctly?
    evolution_brier * 0.30 +         # Are scores well-calibrated?
    pattern_accuracy * 0.20          # Do scores match pattern zones?
)
```

---

## Practical Example: Complete Validation

### Step-by-Step for Alert B

**Day 0 (T+0)**:
```
1. SOT provides ACTUAL features:
   - degree_total: 45
   - total_volume_usd: $250K
   - is_mixer_like: True
   - behavioral_anomaly_score: 0.78

2. Miner analyzes and predicts:
   - score: 0.95 (VERY HIGH RISK)
```

**Day 30 (T+30)**:
```
3. SOT provides new ACTUAL features:
   - degree_total: 180 (+300%)
   - total_volume_usd: $2.5M (+900%)
   - is_mixer_like: True (stayed)
   - behavioral_anomaly_score: 0.95 (+0.17)

4. Validator calculates pattern:
   - Pattern: expanding_illicit (all thresholds exceeded)

5. Validator calculates true probability:
   - true_prob = f(300% degree, 900% volume, mixer, +0.17 anomaly)
   - true_prob = 1.0 (definitely suspicious)

6. Validator evaluates miner:
   - Classification: 0.95 in range (0.70, 1.00) ✅
   - AUC contribution: score 0.95 for label 1 ✅
   - Brier: (0.95 - 1.0)² = 0.0025 ✅ (excellent!)

7. Result:
   - Classification: CORRECT
   - Ranking: Perfect contribution to AUC
   - Calibration: Excellent (Brier near 0)
```

---

## Final Answer

### How Scores Correlate with Features

**The Correlation is Established Through**:

1. **Feature Evolution Patterns** (observed blockchain behavior)
   - expanding_illicit, benign_indicators, dormant
   - Classified using ACTUAL feature thresholds

2. **Pattern → Probability Mapping** (reference point)
   - expanding_illicit → 1.0 (definitely risky)
   - benign_indicators → 0.0 (definitely safe)
   - dormant → 0.2 (probably safe)

3. **Score Validation Metrics**:
   - **AUC**: Are high scores for bad patterns? (ranking)
   - **Brier**: Are scores close to true probabilities? (calibration)
   - **Accuracy**: Are scores in expected ranges? (classification)

### The Complete Formula

```
Tier 3B Score = 
    AUC * 0.50 +           # Ranking quality (order matters)
    (1 - Brier) * 0.30 +   # Calibration (exact values matter)
    Accuracy * 0.20        # Classification (zones matter)

Where:
  - AUC measures: Can you rank threats correctly?
  - Brier measures: Are your confidence levels accurate?
  - Accuracy measures: Are scores in right ranges?
```

### Example Scores

**Good Miner**:
- AUC: 0.95 (excellent ranking)
- Brier: 0.02 → 0.98 (excellent calibration)
- Accuracy: 0.90 (90% in correct zones)
- **Tier 3B = 0.95×0.50 + 0.98×0.30 + 0.90×0.20 = 0.949**

**Medium Miner**:
- AUC: 0.75 (decent ranking)
- Brier: 0.15 → 0.85 (OK calibration)
- Accuracy: 0.70 (70% in correct zones)
- **Tier 3B = 0.75×0.50 + 0.85×0.30 + 0.70×0.20 = 0.770**

**Bad Miner**:
- AUC: 0.55 (barely better than random)
- Brier: 0.25 → 0.75 (poor calibration)
- Accuracy: 0.50 (50% in correct zones)
- **Tier 3B = 0.55×0.50 + 0.75×0.30 + 0.50×0.20 = 0.600**

---

## Conclusion

### The Correlation EXISTS Through:

1. **Feature patterns** (measured from blockchain) → **True probabilities** (calculated)
2. **Miner scores** (predicted) → **Validated against true probabilities**
3. **Three metrics** (AUC, Brier, Accuracy) → **Capture different aspects of quality**

### Why This Works:

✅ **AUC**: Validates ORDERING (high scores for bad patterns)
✅ **Brier**: Validates CALIBRATION (accurate confidence levels)  
✅ **Accuracy**: Validates CLASSIFICATION (scores in right zones)

Together, these three metrics establish the complete correlation between risk scores and feature evolution!

**The punkt odniesienia (reference points) are**:
- Feature evolution thresholds (200%, 300%, etc.)
- Pattern classifications (expanding_illicit, benign, dormant)
- True probability mappings (1.0, 0.0, 0.2)
- Validation metrics (AUC, Brier, Accuracy)

The miner's score is validated against what the ACTUAL blockchain behavior indicates the risk SHOULD be!