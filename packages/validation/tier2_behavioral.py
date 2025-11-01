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
        submitter_id: str,
        processing_date: str,
        window_days: int
    ) -> Dict[str, float]:
        
        logger.info(f"Running Tier 2 behavioral validation for miner {submitter_id}")
        
        with self.client_factory.client_context() as client:
            distribution_entropy = self._check_distribution_entropy(
                client, submitter_id, processing_date, window_days
            )
            
            rank_correlation = self._check_rank_correlation(
                client, submitter_id, processing_date, window_days
            )
            
            consistency_score = self._check_consistency(
                client, submitter_id, processing_date, window_days
            )
            
            address_consistency = self._check_address_consistency(
                client, submitter_id, processing_date, window_days
            )
        
        behavior_score = (
            distribution_entropy * 0.25 +
            rank_correlation * 0.25 +
            consistency_score * 0.25 +
            address_consistency * 0.25
        )
        
        logger.info(f"Tier 2 score for miner {submitter_id}: {behavior_score:.4f}")
        
        return {
            'tier2_behavior_score': behavior_score,
            'tier2_distribution_entropy': distribution_entropy,
            'tier2_rank_correlation': rank_correlation,
            'tier2_consistency_score': consistency_score,
            'tier2_address_consistency': address_consistency
        }
    
    def _check_distribution_entropy(
        self, client, submitter_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        query = """
            SELECT score
            FROM submissions
            WHERE submitter_id = {submitter_id:String}
              AND processing_date = {processing_date:Date}
              AND window_days = {window_days:UInt32}
        """
        
        result = client.query(query, parameters={
            'submitter_id': submitter_id,
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
        self, client, submitter_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        miner_query = """
            SELECT alert_id, score
            FROM submissions
            WHERE submitter_id = {submitter_id:String}
              AND processing_date = {processing_date:Date}
              AND window_days = {window_days:UInt32}
            ORDER BY alert_id
        """
        
        miner_result = client.query(miner_query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        if not miner_result.result_rows:
            return 0.0
        
        miner_scores = {row[0]: row[1] for row in miner_result.result_rows}
        alert_ids = list(miner_scores.keys())
        
        consensus_query = """
            SELECT alert_id, median(score) as median_score
            FROM submissions
            WHERE processing_date = {processing_date:Date}
              AND window_days = {window_days:UInt32}
              AND alert_id IN {alert_ids:Array(String)}
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
        self, client, submitter_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        history_query = """
            SELECT processing_date, COUNT(*) as count
            FROM submissions
            WHERE submitter_id = {submitter_id:String}
              AND window_days = {window_days:UInt32}
              AND processing_date < {processing_date:Date}
            GROUP BY processing_date
            ORDER BY processing_date DESC
            LIMIT 1
        """
        
        history_result = client.query(history_query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        if not history_result.result_rows:
            return 0.7
        
        prev_date = str(history_result.result_rows[0][0])
        
        current_query = """
            SELECT alert_id, score
            FROM submissions
            WHERE submitter_id = {submitter_id:String}
              AND processing_date = {processing_date:Date}
              AND window_days = {window_days:UInt32}
        """
        
        prev_query = """
            SELECT alert_id, score
            FROM submissions
            WHERE submitter_id = {submitter_id:String}
              AND processing_date = {prev_date:Date}
              AND window_days = {window_days:UInt32}
        """
        
        current_result = client.query(current_query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        prev_result = client.query(prev_query, parameters={
            'submitter_id': submitter_id,
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
    
    def _check_address_consistency(
        self, client, submitter_id: str,
        processing_date: str, window_days: int
    ) -> float:
        query = """
            SELECT 
                ra.address,
                ms.score
            FROM submissions ms
            INNER JOIN raw_alerts ra
                ON ms.alert_id = ra.alert_id
                AND ms.processing_date = ra.processing_date
                AND ms.window_days = ra.window_days
            WHERE ms.submitter_id = {submitter_id:String}
              AND ms.processing_date = {processing_date:Date}
              AND ms.window_days = {window_days:UInt32}
        """
        
        result = client.query(query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        if not result.result_rows:
            return 0.0
        
        address_scores = {}
        for row in result.result_rows:
            address, score = row
            if address not in address_scores:
                address_scores[address] = []
            address_scores[address].append(float(score))
        
        penalties = []
        for address, scores in address_scores.items():
            if len(scores) > 1:
                std_dev = np.std(scores)
                penalty = self._calculate_consistency_penalty(std_dev)
                penalties.append(penalty)
        
        if not penalties:
            return 1.0
        
        avg_penalty = np.mean(penalties)
        consistency_score = 1.0 + avg_penalty
        
        return max(0.0, consistency_score)
    
    def _calculate_consistency_penalty(self, std_dev: float) -> float:
        if std_dev < 0.10:
            return 0.0
        elif std_dev < 0.15:
            return -0.05
        elif std_dev < 0.25:
            return -0.10
        else:
            return -0.15