from loguru import logger
from packages.storage import ClientFactory
from typing import Dict
from sklearn.metrics import roc_auc_score, brier_score_loss
import numpy as np


class GroundTruthValidator:
    
    def __init__(self, client_factory: ClientFactory):
        self.client_factory = client_factory
    
    def validate(
        self,
        miner_id: str,
        processing_date: str,
        window_days: int
    ) -> Dict[str, float]:
        
        logger.info(f"Running Tier 3A ground truth validation for miner {miner_id}")
        
        with self.client_factory.client_context() as client:
            scores_and_labels = self._get_scores_with_labels(
                client, miner_id, processing_date, window_days
            )
            
            if not scores_and_labels:
                logger.warning(f"No labeled data for miner {miner_id}")
                return {
                    'tier3_gt_score': None,
                    'tier3_gt_auc': None,
                    'tier3_gt_brier': None,
                    'tier3_gt_coverage': 0.0
                }
            
            miner_scores = scores_and_labels['scores']
            true_labels = scores_and_labels['labels']
            total_alerts = scores_and_labels['total_alerts']
            labeled_count = len(miner_scores)
            
            auc = roc_auc_score(true_labels, miner_scores)
            brier = brier_score_loss(true_labels, miner_scores)
            
            gt_score = auc
            
            coverage = labeled_count / total_alerts if total_alerts > 0 else 0.0
            
            logger.info(
                f"Tier 3A for miner {miner_id}: "
                f"AUC={auc:.4f}, Brier={brier:.4f}, Coverage={coverage:.2%}"
            )
            
            return {
                'tier3_gt_score': gt_score,
                'tier3_gt_auc': auc,
                'tier3_gt_brier': brier,
                'tier3_gt_coverage': coverage
            }
    
    def _get_scores_with_labels(
        self, client, miner_id: str, processing_date: str, window_days: int
    ) -> Dict:
        
        total_query = """
            SELECT COUNT(DISTINCT alert_id)
            FROM miner_submissions
            WHERE miner_id = %(miner_id)s
              AND processing_date = %(processing_date)s
              AND window_days = %(window_days)s
        """
        
        total_result = client.query(total_query, parameters={
            'miner_id': miner_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        total_alerts = total_result.result_rows[0][0]
        
        query = """
            SELECT 
                ms.alert_id,
                ms.score,
                al.risk_level
            FROM miner_submissions ms
            INNER JOIN raw_alerts ra 
                ON ms.alert_id = ra.alert_id 
                AND ms.processing_date = ra.processing_date
                AND ms.window_days = ra.window_days
            INNER JOIN raw_address_labels al
                ON ra.address = al.address
                AND ra.processing_date = al.processing_date
                AND ra.window_days = al.window_days
            WHERE ms.miner_id = %(miner_id)s
              AND ms.processing_date = %(processing_date)s
              AND ms.window_days = %(window_days)s
              AND al.risk_level IN ('low', 'medium', 'high', 'critical')
        """
        
        result = client.query(query, parameters={
            'miner_id': miner_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        if not result.result_rows:
            return None
        
        scores = []
        labels = []
        
        for row in result.result_rows:
            alert_id, score, risk_level = row
            scores.append(float(score))
            
            label = 1 if risk_level in ['high', 'critical'] else 0
            labels.append(label)
        
        return {
            'scores': np.array(scores),
            'labels': np.array(labels),
            'total_alerts': total_alerts
        }