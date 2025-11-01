from loguru import logger
from packages.storage import ClientFactory
from typing import Dict


class IntegrityValidator:
    
    def __init__(self, client_factory: ClientFactory):
        self.client_factory = client_factory
    
    def validate(
        self,
        submitter_id: str,
        processing_date: str,
        window_days: int
    ) -> Dict[str, float]:
        
        logger.info(f"Running Tier 1 integrity validation for miner {submitter_id}")
        
        with self.client_factory.client_context() as client:
            has_all_alerts = self._check_completeness(
                client, submitter_id, processing_date, window_days
            )
            
            score_range_valid = self._check_score_range(
                client, submitter_id, processing_date, window_days
            )
            
            no_duplicates = self._check_duplicates(
                client, submitter_id, processing_date, window_days
            )
            
            metadata_valid = self._check_metadata(
                client, submitter_id, processing_date, window_days
            )
        
        integrity_score = (
            has_all_alerts * 0.25 +
            score_range_valid * 0.25 +
            no_duplicates * 0.25 +
            metadata_valid * 0.25
        )
        
        logger.info(f"Tier 1 score for miner {submitter_id}: {integrity_score:.4f}")
        
        return {
            'tier1_integrity_score': integrity_score,
            'tier1_has_all_alerts': int(has_all_alerts == 1.0),
            'tier1_score_range_valid': int(score_range_valid == 1.0),
            'tier1_no_duplicates': int(no_duplicates == 1.0),
            'tier1_metadata_valid': int(metadata_valid == 1.0)
        }
    
    def _check_completeness(
        self, client, submitter_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        total_alerts_query = """
            SELECT COUNT(DISTINCT alert_id)
            FROM raw_alerts
            WHERE processing_date = %(processing_date)s
              AND window_days = %(window_days)s
        """
        
        total_result = client.query(total_alerts_query, parameters={
            'processing_date': processing_date,
            'window_days': window_days
        })
        total_alerts = total_result.result_rows[0][0]
        
        if total_alerts == 0:
            raise ValueError(f"No alerts found on {processing_date}")
        
        miner_alerts_query = """
            SELECT COUNT(DISTINCT alert_id)
            FROM submissions
            WHERE submitter_id = %(submitter_id)s
              AND processing_date = %(processing_date)s
              AND window_days = %(window_days)s
        """
        
        miner_result = client.query(miner_alerts_query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        miner_alerts = miner_result.result_rows[0][0]
        
        coverage = miner_alerts / total_alerts
        
        return min(coverage, 1.0)
    
    def _check_score_range(
        self, client, submitter_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN score < 0 OR score > 1 THEN 1 ELSE 0 END) as invalid
            FROM submissions
            WHERE submitter_id = %(submitter_id)s
              AND processing_date = %(processing_date)s
              AND window_days = %(window_days)s
        """
        
        result = client.query(query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        total, invalid = result.result_rows[0]
        
        if total == 0:
            return 0.0
        
        return 1.0 - (invalid / total)
    
    def _check_duplicates(
        self, client, submitter_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        query = """
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT alert_id) as unique_alerts
            FROM submissions
            WHERE submitter_id = %(submitter_id)s
              AND processing_date = %(processing_date)s
              AND window_days = %(window_days)s
        """
        
        result = client.query(query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        total, unique = result.result_rows[0]
        
        if total == 0:
            return 0.0
        
        if total == unique:
            return 1.0
        
        return unique / total
    
    def _check_metadata(
        self, client, submitter_id: str,
        processing_date: str, window_days: int
    ) -> float:
        
        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE
                    WHEN model_version = '' OR model_github_url = ''
                    THEN 1 ELSE 0
                END) as invalid
            FROM submissions
            WHERE submitter_id = %(submitter_id)s
              AND processing_date = %(processing_date)s
              AND window_days = %(window_days)s
        """
        
        result = client.query(query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        total, invalid = result.result_rows[0]
        
        if total == 0:
            return 0.0
        
        return 1.0 - (invalid / total)