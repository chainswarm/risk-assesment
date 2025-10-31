#!/usr/bin/env python3
import argparse
from datetime import datetime
from loguru import logger
from packages.storage import ClientFactory, get_connection_params
from packages.validation import IntegrityValidator, BehavioralValidator, GroundTruthValidator


def validate_miner(
    miner_id: str,
    network: str,
    processing_date: str,
    window_days: int
):
    logger.info(f"Validating miner {miner_id} for {network} on {processing_date}")
    
    connection_params = get_connection_params(network)
    client_factory = ClientFactory(connection_params)
    
    tier1_validator = IntegrityValidator(client_factory)
    tier1_results = tier1_validator.validate(
        miner_id, processing_date, window_days
    )
    
    tier2_validator = BehavioralValidator(client_factory)
    tier2_results = tier2_validator.validate(
        miner_id, processing_date, window_days
    )
    
    tier3a_validator = GroundTruthValidator(client_factory)
    tier3a_results = tier3a_validator.validate(
        miner_id, processing_date, window_days
    )
    
    partial_final_score = (
        tier1_results['tier1_integrity_score'] * 0.2 +
        tier2_results['tier2_behavior_score'] * 0.3
    )
    
    if tier3a_results['tier3_gt_score'] is not None:
        gt_contribution = tier3a_results['tier3_gt_score'] * tier3a_results['tier3_gt_coverage'] * 0.5
        partial_final_score += gt_contribution
    
    with client_factory.client_context() as client:
        data = {
            'miner_id': miner_id,
            'processing_date': processing_date,
            'window_days': window_days,
            **tier1_results,
            **tier2_results,
            **tier3a_results,
            'tier3_evolution_score': None,
            'tier3_evolution_auc': None,
            'tier3_evolution_pattern_accuracy': None,
            'tier3_evolution_coverage': None,
            'final_score': partial_final_score,
            'validation_status': 'partial_tier3a',
            'validated_at': datetime.utcnow()
        }
        
        client.insert('miner_validation_results', [data], column_names=list(data.keys()))
    
    logger.info(f"Validation complete for miner {miner_id}")
    logger.info(f"Tier 1 (Integrity): {tier1_results['tier1_integrity_score']:.4f}")
    logger.info(f"Tier 2 (Behavioral): {tier2_results['tier2_behavior_score']:.4f}")
    if tier3a_results['tier3_gt_score'] is not None:
        logger.info(f"Tier 3A (Ground Truth): {tier3a_results['tier3_gt_score']:.4f} (coverage: {tier3a_results['tier3_gt_coverage']:.2%})")
    else:
        logger.info("Tier 3A (Ground Truth): No labeled data")
    logger.info(f"Partial Final Score: {partial_final_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--miner-id", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--processing-date", required=True)
    parser.add_argument("--window-days", type=int, default=195)
    
    args = parser.parse_args()
    
    validate_miner(
        args.miner_id,
        args.network,
        args.processing_date,
        args.window_days
    )