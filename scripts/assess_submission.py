#!/usr/bin/env python3
import argparse
from datetime import datetime
from loguru import logger
from packages.storage import ClientFactory, get_connection_params
from packages.validation import ScoringCoordinator


def assess_submission(
    submitter_id: str,
    network: str,
    processing_date: str,
    window_days: int,
    submission_date: str = None,
    config_path: str = None
):
    logger.info(f"Assessing submission {submitter_id} for {network} on {processing_date}")
    
    connection_params = get_connection_params(network)
    client_factory = ClientFactory(connection_params)
    
    coordinator = ScoringCoordinator(client_factory, config_path)
    
    result = coordinator.calculate_final_score(
        submitter_id=submitter_id,
        processing_date=processing_date,
        window_days=window_days,
        submission_date=submission_date
    )
    
    if result['status'] == 'rejected':
        logger.error(f"Submission {submitter_id} rejected: {result['reason']}")
        logger.info(f"Final Score: {result['final_score']:.4f}")
        return
    
    with client_factory.client_context() as client:
        tier1_results = result.get('tier1_results', {})
        tier2_results = result.get('tier2_results', {})
        tier3a_results = result.get('tier3a_results', {})
        tier3b_results = result.get('tier3b_results', {})
        
        data = {
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days,
            'final_score': result['final_score'],
            'validation_status': result['validation_status'],
            'days_since_submission': result['days_since_submission'],
            'reward_type': result['reward_type'],
            'validated_at': datetime.utcnow()
        }
        
        if tier1_results:
            data.update(tier1_results)
        if tier2_results:
            data.update(tier2_results)
        if tier3a_results:
            data.update(tier3a_results)
        if tier3b_results:
            data.update(tier3b_results)
        
        client.insert('assessment_results', [data], column_names=list(data.keys()))
    
    logger.info(f"Assessment complete for submission {submitter_id}")
    logger.info(f"Status: {result['status']} ({result['validation_status']})")
    logger.info(f"Reward Type: {result['reward_type']}")
    logger.info(f"Days Since Submission: {result['days_since_submission']}")
    
    if tier1_results:
        tier1_score = tier1_results.get('tier1_integrity_score', 0.0)
        logger.info(f"Tier 1 (Integrity): {tier1_score:.4f}")
    
    if tier2_results:
        tier2_score = tier2_results.get('tier2_behavior_score', 0.0)
        tier2_reward = result.get('tier2_reward', 0.0)
        logger.info(f"Tier 2 (Behavioral): {tier2_score:.4f} (reward: {tier2_reward:.4f})")
    
    tier3_score = result.get('tier3_score')
    if tier3_score is not None:
        logger.info(f"Tier 3 (Accuracy): {tier3_score:.4f}")
        
        if tier3a_results and tier3a_results.get('tier3_gt_score') is not None:
            logger.info(f"  - Ground Truth: {tier3a_results['tier3_gt_score']:.4f} (coverage: {tier3a_results['tier3_gt_coverage']:.2%})")
        
        if tier3b_results and tier3b_results.get('tier3_evolution_score') is not None:
            logger.info(f"  - Evolution: {tier3b_results['tier3_evolution_score']:.4f} (coverage: {tier3b_results['tier3_evolution_coverage']:.2%})")
    
    logger.info(f"Final Score: {result['final_score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submitter-id", required=True, help="Submitter ID to validate")
    parser.add_argument("--network", required=True, help="Network name (e.g., ethereum)")
    parser.add_argument("--processing-date", required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--window-days", type=int, default=195, help="Window days for analysis")
    parser.add_argument("--submission-date", help="Original submission date (YYYY-MM-DD), defaults to processing-date")
    parser.add_argument("--config-path", help="Path to custom validation config JSON file")
    
    args = parser.parse_args()
    
    assess_submission(
        submitter_id=args.submitter_id,
        network=args.network,
        processing_date=args.processing_date,
        window_days=args.window_days,
        submission_date=args.submission_date,
        config_path=args.config_path
    )