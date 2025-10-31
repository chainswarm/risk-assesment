# Miner Scoring Methodology - High-Level Overview

## Executive Summary

The Risk Assessment subnet uses a sophisticated multi-tier validation system to score miners based on the quality and accuracy of their risk assessments. Miners are evaluated through three distinct validation tiers, each measuring different aspects of their performance. The final score is a weighted combination of all tiers, ensuring comprehensive evaluation of both technical quality and predictive accuracy.

---

## Core Principles

### 1. Fair Competition
- All miners start with the same baseline data (SOT - Source of Truth)
- Competitive advantage comes from better algorithms, additional data sources, and custom models
- No miner has privileged access to validation mechanisms

### 2. Comprehensive Evaluation
- Multiple validation tiers ensure balanced assessment
- Both technical quality (data integrity) and prediction quality (accuracy) are measured
- Coverage-weighted scoring ensures fair evaluation even with sparse ground truth data

### 3. Real-World Performance
- Validation uses actual blockchain address labels (ground truth)
- Behavioral pattern tracking captures emerging threats
- Temporal consistency ensures stable, reliable predictions

---

## Submission Process

### Step 1: Daily Data Ingestion
Every day at 1 AM UTC, the validator automatically:
- Downloads fresh data from SOT (Source of Truth)
- Validates data integrity (checksums, completeness)
- Ingests into local ClickHouse database
- Tracks 4 core data types:
  - **Alerts**: Suspicious blockchain activity events
  - **Features**: 98 calculated metrics per address
  - **Clusters**: Groups of related addresses
  - **Address Labels**: Known good/bad address classifications

### Step 2: Miner Submission
Miners process the data and submit their assessments via API:
- **Alert Scores**: Risk scores for each alert (0.0 to 1.0)
- **Alert Rankings**: Prioritized list of most suspicious alerts
- **Metadata**: Model version, processing time, GitHub repository link

Each submission includes:
- `miner_id`: Unique identifier for the miner
- `processing_date`: Date of the data being scored
- `window_days`: Data window size (typically 195 days)
- `scores`: Array of alert_id → risk_score mappings

### Step 3: Validation Execution
The validator runs comprehensive validation:
- Triggered manually or on schedule
- Processes all submissions for a given date
- Executes 3-tier validation pipeline
- Stores detailed results and final scores

---

## Three-Tier Validation System

### Tier 1: Data Integrity Validation (Weight: 20%)

**Purpose**: Ensure technical quality and completeness of submissions

**What It Measures**:

1. **Completeness Check**
   - Are all expected alerts scored?
   - Are there any missing predictions?
   - Coverage percentage calculation

2. **Score Range Validation**
   - All scores must be between 0.0 and 1.0
   - No NaN or infinite values
   - Proper probability distributions

3. **Duplicate Detection**
   - No duplicate alert_id entries
   - Each alert scored exactly once
   - Data consistency verification

4. **Metadata Validation**
   - Model version provided
   - GitHub repository link present
   - Processing timestamp valid

**Sub-Scores**:
- `completeness_score`: Percentage of alerts scored
- `score_range_score`: Compliance with valid ranges
- `duplicate_score`: Absence of duplicates (1.0 = no duplicates)
- `metadata_score`: Metadata quality

**Final Tier 1 Score**:
- Average of all 4 sub-scores
- Perfect score = 1.0 (all requirements met)
- Failing any check significantly impacts score

---

### Tier 2: Behavioral Validation (Weight: 30%)

**Purpose**: Assess statistical quality and consistency of predictions

**What It Measures**:

1. **Distribution Entropy**
   - Measures diversity of risk scores
   - Prevents all-same or all-random predictions
   - Ensures meaningful risk stratification
   - Higher entropy (within reason) = better discrimination

2. **Rank Correlation**
   - Compares miner's rankings to baseline SOT rankings
   - Measures alignment with known patterns
   - Spearman correlation coefficient
   - Higher correlation = better consistency with established knowledge

3. **Temporal Consistency**
   - Compares predictions across consecutive days
   - Addresses shouldn't drastically change risk overnight
   - Measures stability and reliability
   - Higher consistency = more trustworthy predictions

**Sub-Scores**:
- `distribution_entropy_score`: Quality of score distribution (0.0 to 1.0)
- `rank_correlation_score`: Alignment with SOT baseline (0.0 to 1.0)
- `temporal_consistency_score`: Day-over-day stability (0.0 to 1.0)

**Final Tier 2 Score**:
- Average of all 3 behavioral sub-scores
- Balances diversity with consistency
- Rewards statistically sound predictions

---

### Tier 3: Predictive Accuracy Validation (Weight: 50%)

**Purpose**: Measure actual prediction quality against real-world data

This tier uses a **hybrid validation approach** combining two complementary methods:

#### Tier 3A: Ground Truth Validation (~10% coverage)

**Data Source**: Known address labels from SOT
- Exchanges (low risk)
- Mixers (high risk)  
- Scams (critical risk)
- Legitimate services (low risk)

**What It Measures**:

1. **AUC (Area Under ROC Curve)**
   - Ability to distinguish high-risk from low-risk addresses
   - Measures discrimination power
   - Range: 0.5 (random) to 1.0 (perfect)
   - Industry standard metric

2. **Brier Score**
   - Measures calibration of probability predictions
   - Rewards well-calibrated probabilities
   - Range: 0.0 (perfect) to 1.0 (worst)
   - Lower is better, inverted for scoring

3. **NDCG (Normalized Discounted Cumulative Gain)**
   - Measures ranking quality
   - Ensures highest-risk alerts ranked first
   - Range: 0.0 to 1.0
   - Higher is better

**Challenge**: Only ~10% of addresses have explicit labels
**Solution**: Combine with Tier 3B for full coverage

---

#### Tier 3B: Feature Evolution Validation (~90% coverage)

**Innovation**: Validates predictions using behavioral patterns over time

**Concept**:
- Track how address features evolve over 30 days
- Suspicious addresses often show specific evolution patterns
- Can validate predictions without explicit labels

**Evolution Tracking Process**:

1. **Baseline Snapshot (T+0)**
   - Capture initial state of all 98 features
   - Record when miner made prediction

2. **Follow-Up Snapshots (T+7, T+14, T+21, T+30)**
   - Track changes in:
     - Transaction degree (connections)
     - Transaction volume (USD)
     - Money in/out flows
     - Graph centrality metrics
     - Behavioral indicators

3. **Pattern Classification**
   - **Expanding Illicit**: Growing connections, increasing volume, suspicious patterns
   - **Benign Indicators**: Stable legitimate patterns, consistent behavior
   - **Dormant**: Little to no activity changes

**Validation Logic**:

If a miner predicted **high risk** AND address shows **expanding illicit patterns**:
- ✅ Correct prediction - Score +1

If a miner predicted **low risk** AND address shows **benign indicators**:
- ✅ Correct prediction - Score +1

If a miner predicted **high risk** BUT address shows **benign indicators**:
- ❌ False positive - Score -1

If a miner predicted **low risk** BUT address shows **expanding illicit patterns**:
- ❌ False negative - Score -1 (more severe penalty)

**Why This Works**:
- Suspicious addresses rarely stay dormant
- Legitimate addresses maintain consistent patterns
- Evolution patterns emerge within 30 days
- No need for explicit labels

---

### Tier 3 Final Calculation: Coverage-Weighted Hybrid

**Formula**:
```
Tier 3 Score = (GT_Score × GT_Coverage) + (Evolution_Score × Evolution_Coverage)
```

**Example**:
- Ground Truth validates 10% of addresses → Coverage = 0.10
- Evolution validates 90% of addresses → Coverage = 0.90
- GT Score = 0.85, Evolution Score = 0.78

```
Tier 3 Score = (0.85 × 0.10) + (0.78 × 0.90)
             = 0.085 + 0.702
             = 0.787
```

**Result**: 100% coverage validation with complementary methods

---

## Final Score Calculation

### Weighted Combination

```
Final Score = (Tier1 × 0.20) + (Tier2 × 0.30) + (Tier3 × 0.50)
```

### Weight Rationale

**Tier 1 (20%)**: Technical baseline
- Must have clean, complete data
- Table stakes for participation
- Lower weight because it's pass/fail

**Tier 2 (30%)**: Statistical quality
- Ensures predictions make statistical sense
- Prevents gaming through randomness
- Critical for reliability

**Tier 3 (50%)**: Predictive accuracy
- Most important: Can you actually predict risk?
- Highest weight because this is the core value
- Combines explicit labels with behavioral patterns

### Example Calculation

**Miner A Performance**:
- Tier 1: 0.95 (excellent data quality)
- Tier 2: 0.82 (good statistical patterns)
- Tier 3: 0.78 (strong predictive power)

```
Final Score = (0.95 × 0.20) + (0.82 × 0.30) + (0.78 × 0.50)
            = 0.190 + 0.246 + 0.390
            = 0.826 (82.6%)
```

**Miner B Performance**:
- Tier 1: 0.98 (near-perfect data)
- Tier 2: 0.65 (mediocre patterns)
- Tier 3: 0.92 (excellent predictions)

```
Final Score = (0.98 × 0.20) + (0.65 × 0.30) + (0.92 × 0.50)
            = 0.196 + 0.195 + 0.460
            = 0.851 (85.1%)
```

**Winner**: Miner B (better predictions despite weaker behavioral patterns)

---

## Ranking System

### Leaderboard Generation

After all miners are scored:

1. **Sort by Final Score** (descending)
2. **Assign Ranks** (1 = highest score)
3. **Handle Ties** (same score = same rank)
4. **Store Rankings** in database

### Public Visibility

Rankings are publicly accessible via API:
- GET `/api/v1/scores/rankings` - All miner rankings
- GET `/api/v1/miners/list` - Miner list with scores
- GET `/api/v1/scores/{miner_id}/latest` - Individual miner details

### Metrics Displayed

For each miner:
- **Final Score**: Overall performance (0.0 to 1.0)
- **Rank**: Position on leaderboard
- **AUC**: Ground truth discrimination
- **Brier**: Probability calibration
- **NDCG**: Ranking quality
- **Model Version**: Which model was used
- **GitHub URL**: Link to miner's repository
- **Status**: Validation status (complete, partial, pending)

---

## Validation Status Categories

### Complete Validation
- All 3 tiers executed successfully
- Both Tier 3A and 3B completed
- Full confidence in final score
- Status: `complete`

### Partial Tier 3A
- Tier 1, 2, and 3B completed
- Tier 3A had insufficient ground truth labels
- Still valid but noted
- Status: `partial_tier3a`

### Tier 3A Only
- Ground truth validation completed
- Evolution tracking not yet available (too recent)
- Valid for immediate scoring
- Status: `tier3a_only`

### Tier 3B Only
- Evolution tracking completed
- No ground truth labels available
- Valid, relies on behavioral patterns
- Status: `tier3b_only`

### No Tier 3
- Only Tier 1 and Tier 2 completed
- Waiting for validation data
- Preliminary score only
- Status: `no_tier3`

---

## Competitive Advantages

### How Miners Improve Their Scores

#### 1. Better Feature Engineering
- Calculate additional risk indicators
- Combine features in novel ways
- Domain-specific metrics

#### 2. Custom Data Sources
- Additional address labels from threat intelligence
- Historical data from multiple sources
- Cross-chain analysis

#### 3. Advanced Models
- Ensemble methods
- Neural networks
- Custom ML algorithms optimized for blockchain data

#### 4. Temporal Analysis
- Time-series modeling
- Trend detection
- Early warning systems

#### 5. Graph Analysis
- Transaction network topology
- Community detection
- Centrality measures

---

## Gaming Prevention

### Anti-Gaming Measures

**1. Multi-Tier Validation**
- Can't optimize for one metric alone
- Must balance technical quality, statistics, and accuracy

**2. Coverage-Weighted Scoring**
- Can't focus only on labeled data
- Must perform well on unlabeled addresses too

**3. Evolution Tracking**
- Predictions validated against future behavior
- Can't game what hasn't happened yet

**4. Baseline Comparison**
- Must outperform SOT baseline
- Pure random predictions score ~0.5

**5. Temporal Consistency**
- Drastic changes penalized
- Encourages stable, reliable models

---

## Transparency & Fairness

### Open Source Validation
- All validation logic is open source
- Miners can inspect scoring methodology
- No hidden evaluation criteria

### Reproducible Results
- All scores stored with metadata
- Validation can be re-run
- Historical performance tracked

### Equal Access
- All miners get same baseline data
- No privileged API access
- Competition based on algorithm quality

### Clear Metrics
- Well-defined scoring criteria
- Industry-standard metrics (AUC, Brier, NDCG)
- Documented weight rationale

---

## Future Enhancements

### Potential Improvements

1. **Dynamic Weights**
   - Adjust tier weights based on data availability
   - Increase Tier 3B weight as more patterns emerge

2. **Network-Specific Scoring**
   - Different weights per blockchain
   - Account for network characteristics

3. **Confidence Intervals**
   - Provide uncertainty estimates
   - Score with error bars

4. **Historical Performance**
   - Long-term consistency rewards
   - Penalize erratic performance

5. **Adversarial Testing**
   - Test against known evasion techniques
   - Robustness scoring

---

## Summary

### Key Takeaways

✅ **Multi-Tier Validation**: Comprehensive evaluation across 3 dimensions

✅ **Hybrid Approach**: Combines ground truth labels (10%) with behavioral evolution (90%)

✅ **Fair Scoring**: Coverage-weighted formula ensures balanced evaluation

✅ **Gaming Resistant**: Multiple validation layers prevent single-metric optimization

✅ **Transparent**: Open source methodology, clear metrics, reproducible results

✅ **Competitive**: Rewards innovation in features, models, and data sources

### Scoring Weights Summary

| Tier | Weight | Focus | Key Metrics |
|------|--------|-------|-------------|
| Tier 1 | 20% | Data Quality | Completeness, validity, metadata |
| Tier 2 | 30% | Statistical Quality | Entropy, correlation, consistency |
| Tier 3 | 50% | Predictive Accuracy | AUC, Brier, NDCG, evolution patterns |

### Validation Coverage

| Method | Coverage | Data Source | Timeframe |
|--------|----------|-------------|-----------|
| Ground Truth (3A) | ~10% | Address labels | Immediate |
| Evolution (3B) | ~90% | Behavioral patterns | 30 days |
| **Total** | **100%** | **Hybrid** | **Varied** |

---

## Conclusion

The Risk Assessment subnet scoring system provides a fair, comprehensive, and gaming-resistant evaluation of miner performance. By combining multiple validation tiers with a hybrid ground truth + behavioral evolution approach, the system achieves 100% coverage validation while maintaining high standards for data quality, statistical soundness, and predictive accuracy.

Miners compete on the quality of their risk assessment algorithms, with success determined by their ability to accurately identify suspicious blockchain activity using both labeled and unlabeled data. The transparent, open-source methodology ensures all miners understand the rules and can optimize their approaches accordingly.

The result is a robust, reliable ranking system that identifies the best risk assessment algorithms and rewards innovation in blockchain security.