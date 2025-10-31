CREATE TABLE IF NOT EXISTS feature_evolution_tracking (
    alert_id String,
    address String,
    base_date Date,
    snapshot_date Date,
    window_days UInt16,
    
    degree_delta Int32,
    in_degree_delta Int32,
    out_degree_delta Int32,
    volume_delta Decimal128(18),
    total_in_usd_delta Decimal128(18),
    total_out_usd_delta Decimal128(18),
    
    pattern_classification String,
    evolution_score Float64,
    
    tracked_at DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (base_date, alert_id, snapshot_date)
SETTINGS index_granularity = 8192;