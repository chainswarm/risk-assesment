from datetime import datetime
from typing import Dict, Any
from loguru import logger
from packages.storage import ClientFactory
from packages.validation.config import ValidationConfig
from packages.validation.tier1_integrity import IntegrityValidator
from packages.validation.tier2_behavioral import BehavioralValidator
from packages.validation.tier3a_ground_truth import GroundTruthValidator
from packages.validation.tier3b_evolution import EvolutionValidator


class ScoringCoordinator:
    
    def __init__(self, client_factory: ClientFactory, config_path: str = None):
        self.client_factory = client_factory
        self.config = ValidationConfig(config_path)
        
        self.tier1_validator = IntegrityValidator(client_factory)
        self.tier2_validator = BehavioralValidator(client_factory)
        self.tier3a_validator = GroundTruthValidator(client_factory)
        self.tier3b_validator = EvolutionValidator(client_factory)
    
    def calculate_final_score(
        self,
        submitter_id: str,
        processing_date: str,
        window_days: int,
        submission_date: str = None
    ) -> Dict[str, Any]:
        logger.info(f"Calculating final score for miner {submitter_id} on {processing_date}")
        
        if submission_date is None:
            submission_date = processing_date
        
        days_since_submission = (
            datetime.strptime(processing_date, '%Y-%m-%d') - 
            datetime.strptime(submission_date, '%Y-%m-%d')
        ).days
        
        if not self.config.tier1_enabled:
            logger.warning("Tier 1 is disabled, skipping")
            tier1_passed = True
            tier1_results = {}
        else:
            tier1_results = self.tier1_validator.validate(
                submitter_id, processing_date, window_days
            )
            tier1_passed = self._check_tier1_gate(tier1_results)
            
            if self.config.tier1_is_gate and not tier1_passed:
                logger.warning(f"Miner {submitter_id} failed Tier 1 gate")
                return {
                    "status": "rejected",
                    "reason": "tier1_failed",
                    "final_score": 0.0,
                    "tier1_passed": False,
                    "tier1_results": tier1_results,
                    "days_since_submission": days_since_submission,
                    "config_snapshot": self.config.get_config_snapshot()
                }
        
        if not self.config.tier2_enabled:
            logger.warning("Tier 2 is disabled, skipping")
            tier2_score = 0.0
            tier2_results = {}
        else:
            tier2_results = self.tier2_validator.validate(
                submitter_id, processing_date, window_days
            )
            tier2_score = tier2_results['tier2_behavior_score']
            
            if tier2_score < self.config.tier2_minimum_score:
                logger.warning(
                    f"Miner {submitter_id} below Tier 2 minimum threshold: "
                    f"{tier2_score:.4f} < {self.config.tier2_minimum_score}"
                )
                if self.config.tier2_is_gate:
                    return {
                        "status": "rejected",
                        "reason": "tier2_below_minimum",
                        "final_score": 0.0,
                        "tier1_passed": True,
                        "tier1_results": tier1_results,
                        "tier2_score": tier2_score,
                        "tier2_results": tier2_results,
                        "tier2_minimum": self.config.tier2_minimum_score,
                        "days_since_submission": days_since_submission,
                        "config_snapshot": self.config.get_config_snapshot()
                    }
        
        tier2_reward = self.config.tier2_flat_reward
        
        if days_since_submission < self.config.flat_period_days:
            logger.info(
                f"Within flat period ({days_since_submission} < {self.config.flat_period_days} days), "
                f"using flat reward: {tier2_reward}"
            )
            final_score = tier2_reward
            
            return {
                "status": "accepted",
                "final_score": final_score,
                "tier1_passed": True,
                "tier1_results": tier1_results,
                "tier2_score": tier2_score,
                "tier2_results": tier2_results,
                "tier2_reward": tier2_reward,
                "tier3_score": None,
                "tier3a_results": None,
                "tier3b_results": None,
                "reward_type": "flat",
                "days_since_submission": days_since_submission,
                "validation_status": "flat_period",
                "config_snapshot": self.config.get_config_snapshot()
            }
        
        if not self.config.tier3_enabled:
            logger.warning("Tier 3 is disabled, using flat reward only")
            final_score = tier2_reward
            
            return {
                "status": "accepted",
                "final_score": final_score,
                "tier1_passed": True,
                "tier1_results": tier1_results,
                "tier2_score": tier2_score,
                "tier2_results": tier2_results,
                "tier2_reward": tier2_reward,
                "tier3_score": None,
                "tier3a_results": None,
                "tier3b_results": None,
                "reward_type": "tier3_disabled",
                "days_since_submission": days_since_submission,
                "validation_status": "tier3_disabled",
                "config_snapshot": self.config.get_config_snapshot()
            }
        
        logger.info(
            f"Past flat period ({days_since_submission} >= {self.config.flat_period_days} days), "
            "calculating accuracy-based reward"
        )
        
        tier3a_results = self.tier3a_validator.validate(
            submitter_id, processing_date, window_days
        )
        
        tier3b_results = {}
        if days_since_submission >= self.config.tier3b_required_days:
            tier3b_results = self.tier3b_validator.validate(
                submitter_id, processing_date, window_days
            )
        else:
            logger.info(
                f"Not enough days for Tier 3B evolution validation: "
                f"{days_since_submission} < {self.config.tier3b_required_days} days"
            )
            tier3b_results = {
                'tier3_evolution_score': None,
                'tier3_evolution_coverage': 0.0
            }
        
        tier3_score = self._calculate_tier3_score(tier3a_results, tier3b_results)
        
        final_score = (
            tier2_score * self.config.tier2_weight +
            tier3_score * self.config.tier3_weight
        )
        
        validation_status = self._determine_validation_status(
            tier3a_results, tier3b_results, days_since_submission
        )
        
        logger.info(f"Final score for miner {submitter_id}: {final_score:.4f}")
        
        return {
            "status": "accepted",
            "final_score": final_score,
            "tier1_passed": True,
            "tier1_results": tier1_results,
            "tier2_score": tier2_score,
            "tier2_results": tier2_results,
            "tier2_reward": tier2_reward,
            "tier3_score": tier3_score,
            "tier3a_results": tier3a_results,
            "tier3b_results": tier3b_results,
            "reward_type": "accuracy_based",
            "days_since_submission": days_since_submission,
            "validation_status": validation_status,
            "config_snapshot": self.config.get_config_snapshot()
        }
    
    def _check_tier1_gate(self, tier1_results: Dict) -> bool:
        integrity_score = tier1_results.get('tier1_integrity_score', 0.0)
        has_all_alerts = tier1_results.get('tier1_has_all_alerts', 0)
        score_range_valid = tier1_results.get('tier1_score_range_valid', 0)
        no_duplicates = tier1_results.get('tier1_no_duplicates', 0)
        metadata_valid = tier1_results.get('tier1_metadata_valid', 0)
        
        all_checks_passed = (
            has_all_alerts == 1 and
            score_range_valid == 1 and
            no_duplicates == 1 and
            metadata_valid == 1
        )
        
        return all_checks_passed
    
    def _calculate_tier3_score(
        self,
        tier3a_results: Dict,
        tier3b_results: Dict
    ) -> float:
        tier3_score = 0.0
        
        tier3a_score = tier3a_results.get('tier3_gt_score')
        tier3a_coverage = tier3a_results.get('tier3_gt_coverage', 0.0)
        
        tier3b_score = tier3b_results.get('tier3_evolution_score')
        tier3b_coverage = tier3b_results.get('tier3_evolution_coverage', 0.0)
        
        if tier3a_score is not None:
            tier3a_contribution = (
                tier3a_score * tier3a_coverage * self.config.tier3a_coverage_weight
            )
            tier3_score += tier3a_contribution
            logger.debug(
                f"Tier 3A contribution: {tier3a_contribution:.4f} "
                f"(score={tier3a_score:.4f}, coverage={tier3a_coverage:.2%})"
            )
        
        if tier3b_score is not None:
            tier3b_contribution = (
                tier3b_score * tier3b_coverage * self.config.tier3b_coverage_weight
            )
            tier3_score += tier3b_contribution
            logger.debug(
                f"Tier 3B contribution: {tier3b_contribution:.4f} "
                f"(score={tier3b_score:.4f}, coverage={tier3b_coverage:.2%})"
            )
        
        if tier3a_score is None and tier3b_score is None:
            logger.warning("No Tier 3 validation available, score = 0.0")
            tier3_score = 0.0
        
        return tier3_score
    
    def _determine_validation_status(
        self,
        tier3a_results: Dict,
        tier3b_results: Dict,
        days_since_submission: int
    ) -> str:
        tier3a_score = tier3a_results.get('tier3_gt_score')
        tier3b_score = tier3b_results.get('tier3_evolution_score')
        
        if tier3a_score is not None and tier3b_score is not None:
            return 'complete'
        elif tier3a_score is None and tier3b_score is None:
            if days_since_submission < self.config.tier3b_required_days:
                return 'awaiting_evolution_data'
            else:
                return 'no_tier3'
        elif tier3a_score is None:
            return 'tier3b_only'
        else:
            return 'tier3a_only'