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