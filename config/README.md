# Validation Configuration System

This directory contains the configurable validation tier weighting system for participant scoring.

## Configuration File

The [`validation_config.json`](validation_config.json) file controls the behavior of the three-tier validation system.

## Configuration Structure

### Tier 1: Integrity (Gate)

```json
"tier1_integrity": {
  "enabled": true,
  "is_gate": true,
  "weight": 0.0
}
```

- **enabled**: Enable/disable tier 1 validation
- **is_gate**: If true, submitters must pass all checks to proceed (fail-fast)
- **weight**: Scoring weight (0.0 means no scoring reward, just gate function)

**Checks Performed:**
- Completeness (all alerts scored)
- Score range validation (0.0-1.0)
- No duplicates
- Valid metadata

### Tier 2: Behavioral (Minimum Threshold)

```json
"tier2_behavioral": {
  "enabled": true,
  "is_gate": false,
  "weight": 0.1,
  "minimum_score": 0.5,
  "flat_reward": 0.1
}
```

- **enabled**: Enable/disable tier 2 validation
- **is_gate**: If true, submitters below minimum_score are rejected
- **weight**: Scoring weight for accuracy-based rewards (after flat period)
- **minimum_score**: Minimum behavioral score required (0.0-1.0)
- **flat_reward**: Flat reward given during flat period (days 0-29)

**Checks Performed:**
- Distribution entropy
- Rank correlation
- Temporal consistency
- Address consistency

### Tier 3: Accuracy (Primary Reward)

```json
"tier3_accuracy": {
  "enabled": true,
  "is_gate": false,
  "weight": 0.9,
  "tier3a_ground_truth": {
    "weight": 0.1,
    "coverage_weight": 0.5
  },
  "tier3b_evolution": {
    "weight": 0.8,
    "coverage_weight": 0.5,
    "required_days": 30
  }
}
```

- **enabled**: Enable/disable tier 3 validation
- **weight**: Overall tier 3 weight in final score
- **tier3a_ground_truth**: Ground truth validation settings
  - **weight**: Relative weight within tier 3
  - **coverage_weight**: Weight factor for coverage calculation
- **tier3b_evolution**: Behavioral evolution validation settings
  - **weight**: Relative weight within tier 3
  - **coverage_weight**: Weight factor for coverage calculation
  - **required_days**: Minimum days since submission for evolution data (default: 30)

### Reward Schedule

```json
"reward_schedule": {
  "flat_period_days": 29,
  "flat_reward": 0.1
}
```

- **flat_period_days**: Number of days to use flat reward (before tier 3 data available)
- **flat_reward**: Reward amount during flat period

## Temporal Logic

The system implements time-based reward progression:

### Days 0-29 (Flat Period)

- **Reward**: Flat reward (0.1) if tier 1 passed and tier 2 >= minimum
- **No Tier 3**: Evolution data not available yet
- **Purpose**: Immediate feedback and base reward

```
Final Score = tier2_flat_reward (if tier1 passed and tier2 >= minimum)
            = 0.1
```

### Day 30+ (Accuracy Period)

- **Reward**: Full accuracy-based scoring
- **Tier 3 Available**: Evolution tracking completed (30 days of data)
- **Purpose**: Reward predictive accuracy

```
Final Score = (tier2_score × tier2_weight) + (tier3_score × tier3_weight)
            = (tier2_score × 0.1) + (tier3_score × 0.9)
```

## Score Calculation

### Tier 3 Score Calculation

```python
tier3_score = (
    tier3a_score × tier3a_coverage × tier3a_coverage_weight +
    tier3b_score × tier3b_coverage × tier3b_coverage_weight
)
```

**Example:**
- Ground truth coverage: 10% → tier3a_score × 0.10 × 0.5
- Evolution coverage: 90% → tier3b_score × 0.90 × 0.5

## Usage

### Basic Validation

```bash
python scripts/assess_submission.py \
  --submitter-id submitter_abc123 \
  --network ethereum \
  --processing-date 2025-11-01 \
  --window-days 195
```

### With Custom Configuration

```bash
python scripts/assess_submission.py \
  --submitter-id submitter_abc123 \
  --network ethereum \
  --processing-date 2025-11-01 \
  --window-days 195 \
  --config-path /path/to/custom_config.json
```

### With Submission Date (for temporal logic)

```bash
python scripts/assess_submission.py \
  --submitter-id submitter_abc123 \
  --network ethereum \
  --processing-date 2025-11-01 \
  --submission-date 2025-10-01 \
  --window-days 195
```

## Validation Status Codes

- **complete**: All tiers executed successfully
- **tier3a_only**: Only ground truth available (no evolution data yet)
- **tier3b_only**: Only evolution available (no ground truth labels)
- **no_tier3**: No tier 3 data available
- **flat_period**: Within flat reward period (days 0-29)
- **awaiting_evolution_data**: Waiting for 30-day evolution tracking

## Rejection Reasons

- **tier1_failed**: Failed integrity gate checks
- **tier2_below_minimum**: Behavioral score below minimum threshold

## Customization

To customize the validation system, edit [`validation_config.json`](validation_config.json):

1. Adjust tier weights to change scoring priorities
2. Modify `flat_period_days` to change when tier 3 kicks in
3. Change `tier3b_required_days` to require more/less evolution data
4. Set `tier2_minimum_score` to enforce quality threshold
5. Toggle `is_gate` flags to enable/disable fail-fast behavior

## Configuration Validation

The system validates configuration on load:

- All required keys must be present
- `is_gate` must be boolean
- `minimum_score` must be numeric
- `required_days` must be integer
- Weights should sum logically (tier2 + tier3 weights)

## Examples

### High Accuracy Focus

```json
{
  "tier2_behavioral": {
    "weight": 0.05
  },
  "tier3_accuracy": {
    "weight": 0.95
  }
}
```

### Longer Flat Period

```json
{
  "reward_schedule": {
    "flat_period_days": 60,
    "flat_reward": 0.15
  }
}
```

### Stricter Quality Requirements

```json
{
  "tier2_behavioral": {
    "is_gate": true,
    "minimum_score": 0.7
  }
}
```

## See Also

- [`packages/validation/scoring_coordinator.py`](../packages/validation/scoring_coordinator.py) - Main scoring logic
- [`packages/validation/config.py`](../packages/validation/config.py) - Configuration loader
- [`scripts/assess_submission.py`](../scripts/assess_submission.py) - Validation script