from loguru import logger
from packages.storage import ClientFactory
from typing import Dict, List
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
import numpy as np


class EvolutionValidator:
    
    def __init__(self, client_factory: ClientFactory):
        self.client_factory = client_factory
    
    def track_evolution(
        self,
        base_date: str,
        window_days: int,
        snapshot_days: List[int] = None
    ):
        if snapshot_days is None:
            snapshot_days = [0, 7, 14, 21, 30]
            
        logger.info(f"Tracking feature evolution from {base_date}")
        
        with self.client_factory.client_context() as client:
            alerts_query = """
                SELECT DISTINCT alert_id, address
                FROM raw_alerts
                WHERE processing_date = {base_date:Date}
                  AND window_days = {window_days:UInt32}
            """
            
            alerts_result = client.query(alerts_query, parameters={
                'base_date': base_date,
                'window_days': window_days
            })
            
            if not alerts_result.result_rows:
                logger.warning(f"No alerts found for {base_date}")
                return
            
            alert_addresses = {row[0]: row[1] for row in alerts_result.result_rows}
            
            evolution_data = []
            
            for days_offset in snapshot_days:
                snapshot_date = (datetime.strptime(base_date, '%Y-%m-%d') + 
                               timedelta(days=days_offset)).strftime('%Y-%m-%d')
                
                logger.info(f"Capturing snapshot at T+{days_offset} ({snapshot_date})")
                
                for alert_id, address in alert_addresses.items():
                    deltas = self._calculate_feature_deltas(
                        client, address, base_date, snapshot_date, window_days
                    )
                    
                    if deltas:
                        pattern = self._classify_pattern(deltas)
                        evolution_score = self._calculate_evolution_score(deltas, pattern)
                        
                        evolution_data.append({
                            'alert_id': alert_id,
                            'address': address,
                            'base_date': base_date,
                            'snapshot_date': snapshot_date,
                            'window_days': window_days,
                            **deltas,
                            'pattern_classification': pattern,
                            'evolution_score': evolution_score,
                            'tracked_at': datetime.utcnow()
                        })
            
            if evolution_data:
                client.insert('feature_evolution_tracking', evolution_data, 
                            column_names=list(evolution_data[0].keys()))
                logger.info(f"Tracked {len(evolution_data)} evolution snapshots")
    
    def validate(
        self,
        miner_id: str,
        processing_date: str,
        window_days: int
    ) -> Dict[str, float]:
        logger.info(f"Running Tier 3B evolution validation for miner {miner_id}")
        
        with self.client_factory.client_context() as client:
            scores_and_evolution = self._get_scores_with_evolution(
                client, miner_id, processing_date, window_days
            )
            
            if not scores_and_evolution:
                logger.warning(f"No evolution data for miner {miner_id}")
                return {
                    'tier3_evolution_score': None,
                    'tier3_evolution_auc': None,
                    'tier3_evolution_pattern_accuracy': None,
                    'tier3_evolution_coverage': 0.0
                }
            
            miner_scores = scores_and_evolution['scores']
            evolution_labels = scores_and_evolution['labels']
            pattern_predictions = scores_and_evolution['pattern_predictions']
            total_alerts = scores_and_evolution['total_alerts']
            tracked_count = len(miner_scores)
            
            auc = roc_auc_score(evolution_labels, miner_scores)
            
            pattern_accuracy = np.mean(pattern_predictions)
            
            evolution_score = (auc * 0.7) + (pattern_accuracy * 0.3)
            
            coverage = tracked_count / total_alerts if total_alerts > 0 else 0.0
            
            logger.info(
                f"Tier 3B for miner {miner_id}: "
                f"AUC={auc:.4f}, Pattern Acc={pattern_accuracy:.4f}, "
                f"Evolution Score={evolution_score:.4f}, Coverage={coverage:.2%}"
            )
            
            return {
                'tier3_evolution_score': evolution_score,
                'tier3_evolution_auc': auc,
                'tier3_evolution_pattern_accuracy': pattern_accuracy,
                'tier3_evolution_coverage': coverage
            }
    
    def _calculate_feature_deltas(
        self, client, address: str, base_date: str, 
        snapshot_date: str, window_days: int
    ) -> Dict:
        base_query = """
            SELECT degree, in_degree, out_degree, total_volume_usd, 
                   total_in_usd, total_out_usd
            FROM raw_features
            WHERE address = {address:String}
              AND processing_date = {base_date:Date}
              AND window_days = {window_days:UInt32}
        """
        
        base_result = client.query(base_query, parameters={
            'address': address,
            'base_date': base_date,
            'window_days': window_days
        })
        
        if not base_result.result_rows:
            return None
        
        base_row = base_result.result_rows[0]
        
        snapshot_query = """
            SELECT degree, in_degree, out_degree, total_volume_usd,
                   total_in_usd, total_out_usd
            FROM raw_features
            WHERE address = {address:String}
              AND processing_date = {snapshot_date:Date}
              AND window_days = {window_days:UInt32}
        """
        
        snapshot_result = client.query(snapshot_query, parameters={
            'address': address,
            'snapshot_date': snapshot_date,
            'window_days': window_days
        })
        
        if not snapshot_result.result_rows:
            return None
        
        snapshot_row = snapshot_result.result_rows[0]
        
        return {
            'degree_delta': int(snapshot_row[0] - base_row[0]),
            'in_degree_delta': int(snapshot_row[1] - base_row[1]),
            'out_degree_delta': int(snapshot_row[2] - base_row[2]),
            'volume_delta': float(snapshot_row[3] - base_row[3]),
            'total_in_usd_delta': float(snapshot_row[4] - base_row[4]),
            'total_out_usd_delta': float(snapshot_row[5] - base_row[5])
        }
    
    def _classify_pattern(self, deltas: Dict) -> str:
        degree_growing = deltas['degree_delta'] > 5
        volume_growing = deltas['volume_delta'] > 1000
        
        if degree_growing and volume_growing:
            return 'expanding_illicit'
        elif deltas['degree_delta'] < -2 or deltas['volume_delta'] < -500:
            return 'benign_indicators'
        else:
            return 'dormant'
    
    def _calculate_evolution_score(self, deltas: Dict, pattern: str) -> float:
        if pattern == 'expanding_illicit':
            return 0.9
        elif pattern == 'benign_indicators':
            return 0.1
        else:
            return 0.5
    
    def _get_scores_with_evolution(
        self, client, miner_id: str, processing_date: str, window_days: int
    ) -> Dict:
        total_query = """
            SELECT COUNT(DISTINCT alert_id)
            FROM miner_submissions
            WHERE miner_id = {miner_id:String}
              AND processing_date = {processing_date:Date}
              AND window_days = {window_days:UInt32}
        """
        
        total_result = client.query(total_query, parameters={
            'miner_id': miner_id,
            'processing_date': processing_date,
            'window_days': window_days
        })
        
        total_alerts = total_result.result_rows[0][0]
        
        snapshot_30_date = (datetime.strptime(processing_date, '%Y-%m-%d') + 
                           timedelta(days=30)).strftime('%Y-%m-%d')
        
        query = """
            SELECT 
                ms.alert_id,
                ms.score,
                fe.pattern_classification,
                fe.evolution_score
            FROM miner_submissions ms
            INNER JOIN feature_evolution_tracking fe
                ON ms.alert_id = fe.alert_id
                AND fe.base_date = {processing_date:Date}
                AND fe.snapshot_date = {snapshot_30_date:Date}
                AND fe.window_days = {window_days:UInt32}
            WHERE ms.miner_id = {miner_id:String}
              AND ms.processing_date = {processing_date:Date}
              AND ms.window_days = {window_days:UInt32}
        """
        
        result = client.query(query, parameters={
            'miner_id': miner_id,
            'processing_date': processing_date,
            'snapshot_30_date': snapshot_30_date,
            'window_days': window_days
        })
        
        if not result.result_rows:
            return None
        
        scores = []
        labels = []
        pattern_predictions = []
        
        for row in result.result_rows:
            alert_id, score, pattern, evolution_score = row
            scores.append(float(score))
            
            if pattern == 'expanding_illicit':
                labels.append(1)
                pattern_predictions.append(1 if score > 0.5 else 0)
            elif pattern == 'benign_indicators':
                labels.append(0)
                pattern_predictions.append(1 if score < 0.5 else 0)
            else:
                continue
        
        if not scores:
            return None
        
        return {
            'scores': np.array(scores),
            'labels': np.array(labels),
            'pattern_predictions': np.array(pattern_predictions),
            'total_alerts': total_alerts
        }