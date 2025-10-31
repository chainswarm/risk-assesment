CREATE TABLE IF NOT EXISTS miner_submissions (
    submission_id String,
    miner_id String,
    network String,
    processing_date Date,
    window_days UInt16,
    alert_id String,
    score Float64,
    model_version String,
    model_github_url String,
    submission_timestamp DateTime64(3),
    score_metadata String
) ENGINE = MergeTree()
ORDER BY (network, processing_date, window_days, miner_id, alert_id)
SETTINGS index_granularity = 8192;