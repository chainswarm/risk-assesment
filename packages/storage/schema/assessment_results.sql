CREATE TABLE IF NOT EXISTS assessment_results (
    submitter_id String,
    processing_date Date,
    window_days UInt16,
    
    tier1_integrity_score Float64,
    tier1_has_all_alerts UInt8,
    tier1_score_range_valid UInt8,
    tier1_no_duplicates UInt8,
    tier1_metadata_valid UInt8,
    
    tier2_behavior_score Float64,
    tier2_distribution_entropy Float64,
    tier2_rank_correlation Float64,
    tier2_consistency_score Float64,
    
    tier3_gt_score Float64,
    tier3_gt_auc Float64,
    tier3_gt_brier Float64,
    tier3_gt_coverage Float64,
    
    tier3_evolution_score Float64,
    tier3_evolution_auc Float64,
    tier3_evolution_pattern_accuracy Float64,
    tier3_evolution_coverage Float64,
    
    final_score Float64,
    validation_status String,
    
    validated_at DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (processing_date, window_days, submitter_id)
SETTINGS index_granularity = 8192;