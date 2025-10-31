from loguru import logger
from packages.storage import ClientFactory
from typing import Dict
import numpy as np
from scipy.stats import entropy, spearmanr


class BehavioralValidator:
    
    def __init__(self, client_factory: ClientFactory):
        self.client_factory = client_factory
    
    def validate(
        self,
        miner_id: str,
        processing_date: str,
        window_days: int
    ) -> Dict[str, float]:
        
        logger.info(f"Running Tier 2 behavioral validation for miner {miner_id}")
        
        with self.client_factory.client_context() as client:
            distribution_entropy = self._check_distribution_entropy(
                client, miner_id, processing_date, window_days
            )
            
            rank_correlation = self._check_rank_correlation(
                client, miner_id, processing_date, window_days
            )
            
            consistency_score = self._check_consistency(
                client, miner_id, processing_date, window_days
            )
        
        behavior_score = (
            distribution_entropy * 0.33 +
            rank_correlation * 0.33 +
            consistency_score * 0.34
        )
        
        logger.info(f"Tier 2 score for miner {miner_id}: {behavior_score:.4f}")
        
        return {
            'tier2_behavior_score': behavior_score,
            'tier2_distribution_entropy': distribution_entropy,
            'tier2_rank_correlation': rank_correlation,
            'tier2_consistency_score': consistency_score
        }
    
    def _check_distribution_entropy(
        self, client, miner_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        query = """
            SELECT score
            FROM miner_submissions
            WHERE miner_id = %(miner_id)s
              AND processing_date = %(processing_date)s
              AND window_days = %(window_days)s
        """
        
        result = client.query(query, parameters={
            'miner_id': miner_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        if not result.result_rows:
            return 0.0
        
        scores = [row[0] for row in result.result_rows]
        
        hist, _ = np.histogram(scores, bins=10, range=(0, 1))
        
        hist = hist / len(scores)
        
        score_entropy = entropy(hist + 1e-10)
        
        max_entropy = np.log(10)
        normalized_entropy = score_entropy / max_entropy
        
        return normalized_entropy
    
    def _check_rank_correlation(
        self, client, miner_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        miner_query = """
            SELECT alert_id, score
            FROM miner_submissions
            WHERE miner_id = %(miner_id)s
              AND processing_date = %(processing_date)s
              AND window_days = %(window_days)s
            ORDER BY alert_id
        """
        
        miner_result = client.query(miner_query, parameters={
            'miner_id': miner_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        if not miner_result.result_rows:
            return 0.0
        
        miner_scores = {row[0]: row[1] for row in miner_result.result_rows}
        alert_ids = list(miner_scores.keys())
        
        consensus_query = """
            SELECT alert_id, median(score) as median_score
            FROM miner_submissions
            WHERE processing_date = %(processing_date)s
              AND window_days = %(window_days)s
              AND alert_id IN %(alert_ids)s
            GROUP BY alert_id
            ORDER BY alert_id
        """
        
        consensus_result = client.query(consensus_query, parameters={
            'processing_date': processing_date,
            'window_days': window_days,
            'alert_ids': alert_ids
        })
        
        if not consensus_result.result_rows:
            return 0.0
        
        consensus_scores = {row[0]: row[1] for row in consensus_result.result_rows}
        
        miner_ranks = []
        consensus_ranks = []
        
        for alert_id in alert_ids:
            if alert_id in consensus_scores:
                miner_ranks.append(miner_scores[alert_id])
                consensus_ranks.append(consensus_scores[alert_id])
        
        if len(miner_ranks) < 2:
            return 0.0
        
        correlation, _ = spearmanr(miner_ranks, consensus_ranks)
        
        normalized_correlation = (correlation + 1) / 2
        
        return normalized_correlation
    
    def _check_consistency(
        self, client, miner_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        history_query = """
            SELECT processing_date, COUNT(*) as count
            FROM miner_submissions
            WHERE miner_id = %(miner_id)s
              AND window_days = %(window_days)s
              AND processing_date < %(processing_date)s
            GROUP BY processing_date
            ORDER BY processing_date DESC
            LIMIT 1
        """
        
        history_result = client.query(history_query, parameters={
            'miner_id': miner_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        if not history_result.result_rows:
            return 0.7
        
        prev_date = str(history_result.result_rows[0][0])
        
        current_query = """
            SELECT alert_id, score
            FROM miner_submissions
            WHERE miner_id = %(miner_id)s
              AND processing_date = %(processing_date)s
              AND window_days = %(window_days)s
        """
        
        prev_query = """
            SELECT alert_id, score
            FROM miner_submissions
            WHERE miner_id = %(miner_id)s
              AND processing_date = %(prev_date)s
              AND window_days = %(window_days)s
        """
        
        current_result = client.query(current_query, parameters={
            'miner_id': miner_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        prev_result = client.query(prev_query, parameters={
            'miner_id': miner_id,
            'prev_date': prev_date,
            'window_days': window_days
        })
        
        current_scores = {row[0]: row[1] for row in current_result.result_rows}
        prev_scores = {row[0]: row[1] for row in prev_result.result_rows}
        
        overlap = set(current_scores.keys()) & set(prev_scores.keys())
        
        if len(overlap) < 2:
            return 0.7
        
        differences = []
        for alert_id in overlap:
            diff = abs(current_scores[alert_id] - prev_scores[alert_id])
            differences.append(diff)
        
        mean_diff = np.mean(differences)
        
        consistency = 1.0 - min(mean_diff, 1.0)
        
        return consistency