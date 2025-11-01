from loguru import logger
from packages.storage import ClientFactory
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score, brier_score_loss
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
                        pattern, expected_range = self._classify_pattern(deltas)
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
        submitter_id: str,
        processing_date: str,
        window_days: int
    ) -> Dict[str, float]:
        logger.info(f"Running Tier 3B evolution validation for miner {submitter_id}")
        
        with self.client_factory.client_context() as client:
            alert_details = self._validate_per_alert(
                client, submitter_id, processing_date, window_days
            )
            
            if not alert_details:
                logger.warning(f"No evolution data for miner {submitter_id}")
                return {
                    'tier3_evolution_score': None,
                    'tier3_auc_score': None,
                    'tier3_brier_score': None,
                    'tier3_pattern_accuracy': None,
                    'tier2_consistency_score': None,
                    'tier3_evolution_coverage': 0.0
                }
            
            address_aggregates = self._aggregate_by_address(alert_details)
            
            consistency_score = self._calculate_overall_consistency(address_aggregates)
            
            metrics = self._calculate_validation_metrics(alert_details)
            
            final_score = (
                metrics['auc_score'] * 0.40 +
                (1.0 - metrics['brier_score']) * 0.30 +
                metrics['pattern_accuracy'] * 0.30
            )
            
            self._store_alert_details(
                client, submitter_id, processing_date, window_days, alert_details
            )
            
            total_alerts = self._get_total_alerts(client, submitter_id, processing_date, window_days)
            coverage = len(alert_details) / total_alerts if total_alerts > 0 else 0.0
            
            logger.info(
                f"Tier 3B for miner {submitter_id}: "
                f"AUC={metrics['auc_score']:.4f}, Brier={metrics['brier_score']:.4f}, "
                f"Pattern Acc={metrics['pattern_accuracy']:.4f}, "
                f"Consistency={consistency_score:.4f}, Final={final_score:.4f}"
            )
            
            return {
                'tier3_evolution_score': final_score,
                'tier3_auc_score': metrics['auc_score'],
                'tier3_brier_score': metrics['brier_score'],
                'tier3_pattern_accuracy': metrics['pattern_accuracy'],
                'tier2_consistency_score': consistency_score,
                'tier3_evolution_coverage': coverage
            }
    
    def _validate_per_alert(
        self, client, submitter_id: str, processing_date: str, window_days: int
    ) -> List[Dict]:
        snapshot_30_date = (datetime.strptime(processing_date, '%Y-%m-%d') + 
                           timedelta(days=30)).strftime('%Y-%m-%d')
        
        query = """
            SELECT 
                ms.alert_id,
                ms.score,
                ra.address,
                fe.pattern_classification,
                fe.degree_delta_pct,
                fe.volume_delta_pct,
                fe.evolution_score
            FROM submissions ms
            INNER JOIN raw_alerts ra
                ON ms.alert_id = ra.alert_id
                AND ms.processing_date = ra.processing_date
                AND ms.window_days = ra.window_days
            INNER JOIN feature_evolution_tracking fe
                ON ms.alert_id = fe.alert_id
                AND fe.base_date = {processing_date:Date}
                AND fe.snapshot_date = {snapshot_30_date:Date}
                AND fe.window_days = {window_days:UInt32}
            WHERE ms.submitter_id = {submitter_id:String}
              AND ms.processing_date = {processing_date:Date}
              AND ms.window_days = {window_days:UInt32}
        """
        
        result = client.query(query, parameters={
            'submitter_id': submitter_id,
            'processing_date': processing_date,
            'snapshot_30_date': snapshot_30_date,
            'window_days': window_days
        })
        
        if not result.result_rows:
            return []
        
        alert_details = []
        
        for row in result.result_rows:
            alert_id, score, address, pattern, degree_delta_pct, volume_delta_pct, evolution_score = row
            
            expected_range = self._get_expected_range(pattern)
            
            validation_score = self._validate_single_score(score, expected_range)
            
            pattern_match_score = 1.0 if expected_range[0] <= score <= expected_range[1] else 0.0
            
            alert_details.append({
                'alert_id': alert_id,
                'address': address,
                'submitted_score': float(score),
                'evolution_validation_score': validation_score,
                'pattern_classification': pattern,
                'pattern_match_score': pattern_match_score,
                'evolution_score': float(evolution_score)
            })
        
        return alert_details
    
    def _aggregate_by_address(self, alert_details: List[Dict]) -> List[Dict]:
        address_groups = {}
        
        for detail in alert_details:
            address = detail['address']
            if address not in address_groups:
                address_groups[address] = []
            address_groups[address].append(detail)
        
        address_aggregates = []
        
        for address, details in address_groups.items():
            scores = [d['evolution_validation_score'] for d in details]
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            penalty = self._calculate_consistency_penalty(std_score)
            
            address_aggregates.append({
                'address': address,
                'avg_score': avg_score,
                'std_score': std_score,
                'penalty': penalty,
                'final_score': avg_score + penalty,
                'alert_count': len(details)
            })
        
        return address_aggregates
    
    def _calculate_consistency_penalty(self, std_dev: float) -> float:
        if std_dev < 0.10:
            return 0.0
        elif std_dev < 0.15:
            return -0.05
        elif std_dev < 0.25:
            return -0.10
        else:
            return -0.15
    
    def _calculate_overall_consistency(self, address_aggregates: List[Dict]) -> float:
        if not address_aggregates:
            return 0.0
        
        penalties = [agg['penalty'] for agg in address_aggregates]
        avg_penalty = np.mean(penalties)
        
        consistency_score = 1.0 + avg_penalty
        
        return max(0.0, consistency_score)
    
    def _calculate_validation_metrics(self, alert_details: List[Dict]) -> Dict[str, float]:
        scores = np.array([d['submitted_score'] for d in alert_details])
        validation_scores = np.array([d['evolution_validation_score'] for d in alert_details])
        pattern_matches = np.array([d['pattern_match_score'] for d in alert_details])
        
        labels = []
        for detail in alert_details:
            pattern = detail['pattern_classification']
            if pattern == 'expanding_illicit':
                labels.append(1)
            elif pattern == 'benign_indicators':
                labels.append(0)
            else:
                labels.append(0 if detail['submitted_score'] < 0.5 else 1)
        
        labels = np.array(labels)
        
        if len(np.unique(labels)) < 2:
            auc_score = 0.5
        else:
            auc_score = roc_auc_score(labels, scores)
        
        brier_score = brier_score_loss(labels, scores)
        
        pattern_accuracy = np.mean(pattern_matches)
        
        return {
            'auc_score': auc_score,
            'brier_score': brier_score,
            'pattern_accuracy': pattern_accuracy
        }
    
    def _validate_single_score(self, score: float, expected_range: Tuple[float, float]) -> float:
        if expected_range[0] <= score <= expected_range[1]:
            return 1.0
        else:
            if score < expected_range[0]:
                distance = expected_range[0] - score
            else:
                distance = score - expected_range[1]
            
            match_score = max(0.0, 1.0 - (distance * 2))
            
            return match_score
    
    def _get_expected_range(self, pattern: str) -> Tuple[float, float]:
        if pattern == 'expanding_illicit':
            return (0.70, 1.00)
        elif pattern == 'benign_indicators':
            return (0.00, 0.30)
        elif pattern == 'dormant':
            return (0.15, 0.25)
        else:
            return (0.30, 0.70)
    
    def _store_alert_details(
        self, client, submitter_id: str, processing_date: str, 
        window_days: int, alert_details: List[Dict]
    ):
        if not alert_details:
            return
        
        records = []
        for detail in alert_details:
            records.append({
                'submitter_id': submitter_id,
                'processing_date': processing_date,
                'window_days': window_days,
                'alert_id': detail['alert_id'],
                'address': detail['address'],
                'submitted_score': detail['submitted_score'],
                'evolution_validation_score': detail['evolution_validation_score'],
                'pattern_classification': detail['pattern_classification'],
                'pattern_match_score': detail['pattern_match_score'],
                'validated_at': datetime.utcnow()
            })
        
        client.insert('alert_validation_details', records, 
                     column_names=list(records[0].keys()))
        logger.info(f"Stored {len(records)} alert validation details")
    
    def _get_total_alerts(
        self, client, submitter_id: str, processing_date: str, window_days: int
    ) -> int:
        query = """
            SELECT COUNT(DISTINCT alert_id)
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
        
        return result.result_rows[0][0] if result.result_rows else 0
    
    def _calculate_feature_deltas(
        self, client, address: str, base_date: str, 
        snapshot_date: str, window_days: int
    ) -> Dict:
        base_query = """
            SELECT 
                degree_total, degree_in, degree_out,
                total_volume_usd, total_in_usd, total_out_usd,
                is_mixer_like, is_exchange_like,
                behavioral_anomaly_score, graph_anomaly_score,
                velocity_score, burst_factor,
                pagerank, betweenness_centrality
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
            SELECT 
                degree_total, degree_in, degree_out,
                total_volume_usd, total_in_usd, total_out_usd,
                is_mixer_like, is_exchange_like,
                behavioral_anomaly_score, graph_anomaly_score,
                velocity_score, burst_factor,
                pagerank, betweenness_centrality
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
        
        degree_delta = int(snapshot_row[0] - base_row[0])
        degree_delta_pct = (degree_delta / base_row[0] * 100) if base_row[0] > 0 else 0
        
        volume_delta = float(snapshot_row[3] - base_row[3])
        volume_delta_pct = (volume_delta / base_row[3] * 100) if base_row[3] > 0 else 0
        
        return {
            'degree_delta': degree_delta,
            'degree_delta_pct': degree_delta_pct,
            'in_degree_delta': int(snapshot_row[1] - base_row[1]),
            'out_degree_delta': int(snapshot_row[2] - base_row[2]),
            'volume_delta': volume_delta,
            'volume_delta_pct': volume_delta_pct,
            'total_in_usd_delta': float(snapshot_row[4] - base_row[4]),
            'total_out_usd_delta': float(snapshot_row[5] - base_row[5]),
            'is_mixer_like': snapshot_row[6],
            'is_exchange_like': snapshot_row[7],
            'behavioral_anomaly_score': float(snapshot_row[8]),
            'graph_anomaly_score': float(snapshot_row[9]),
            'velocity_score': float(snapshot_row[10]),
            'burst_factor': float(snapshot_row[11]),
            'pagerank': float(snapshot_row[12]),
            'betweenness_centrality': float(snapshot_row[13])
        }
    
    def _classify_pattern(self, deltas: Dict) -> Tuple[str, Tuple[float, float]]:
        degree_growth = deltas['degree_delta_pct']
        volume_growth = deltas['volume_delta_pct']
        is_mixer = deltas['is_mixer_like']
        anomaly = deltas['behavioral_anomaly_score']
        velocity = deltas['velocity_score']
        burst = deltas['burst_factor']
        
        if (degree_growth > 200 and 
            volume_growth > 300 and
            (is_mixer or anomaly > 0.7 or velocity > 0.8)):
            return 'expanding_illicit', (0.70, 1.00)
        
        if (degree_growth < 50 and 
            volume_growth < 100 and
            anomaly < 0.3 and
            not is_mixer):
            return 'benign_indicators', (0.00, 0.30)
        
        if (degree_growth < 20 and 
            volume_growth < 30 and
            velocity < 0.3):
            return 'dormant', (0.15, 0.25)
        
        return 'ambiguous', (0.30, 0.70)
    
    def _calculate_evolution_score(self, deltas: Dict, pattern: str) -> float:
        if pattern == 'expanding_illicit':
            return 0.9
        elif pattern == 'benign_indicators':
            return 0.1
        elif pattern == 'dormant':
            return 0.2
        else:
            return 0.5