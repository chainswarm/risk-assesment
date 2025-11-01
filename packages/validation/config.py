import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ValidationConfig:
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "validation_config.json"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = json.load(f)
        
        self._validate_config()
        logger.info(f"Loaded validation configuration from {config_path}")
    
    def _validate_config(self):
        required_keys = ['tier1_integrity', 'tier2_behavioral', 'tier3_accuracy', 'reward_schedule']
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        if not isinstance(self._config['tier1_integrity'].get('is_gate'), bool):
            raise ValueError("tier1_integrity.is_gate must be a boolean")
        
        if not isinstance(self._config['tier2_behavioral'].get('minimum_score'), (int, float)):
            raise ValueError("tier2_behavioral.minimum_score must be a number")
        
        if not isinstance(self._config['tier3_accuracy']['tier3b_evolution'].get('required_days'), int):
            raise ValueError("tier3_accuracy.tier3b_evolution.required_days must be an integer")
    
    @property
    def tier1_enabled(self) -> bool:
        return self._config['tier1_integrity'].get('enabled', True)
    
    @property
    def tier1_is_gate(self) -> bool:
        return self._config['tier1_integrity'].get('is_gate', True)
    
    @property
    def tier1_weight(self) -> float:
        return self._config['tier1_integrity'].get('weight', 0.0)
    
    @property
    def tier2_enabled(self) -> bool:
        return self._config['tier2_behavioral'].get('enabled', True)
    
    @property
    def tier2_is_gate(self) -> bool:
        return self._config['tier2_behavioral'].get('is_gate', False)
    
    @property
    def tier2_weight(self) -> float:
        return self._config['tier2_behavioral'].get('weight', 0.1)
    
    @property
    def tier2_minimum_score(self) -> float:
        return self._config['tier2_behavioral'].get('minimum_score', 0.5)
    
    @property
    def tier2_flat_reward(self) -> float:
        return self._config['tier2_behavioral'].get('flat_reward', 0.1)
    
    @property
    def tier3_enabled(self) -> bool:
        return self._config['tier3_accuracy'].get('enabled', True)
    
    @property
    def tier3_weight(self) -> float:
        return self._config['tier3_accuracy'].get('weight', 0.9)
    
    @property
    def tier3a_weight(self) -> float:
        return self._config['tier3_accuracy']['tier3a_ground_truth'].get('weight', 0.1)
    
    @property
    def tier3a_coverage_weight(self) -> float:
        return self._config['tier3_accuracy']['tier3a_ground_truth'].get('coverage_weight', 0.5)
    
    @property
    def tier3b_weight(self) -> float:
        return self._config['tier3_accuracy']['tier3b_evolution'].get('weight', 0.8)
    
    @property
    def tier3b_coverage_weight(self) -> float:
        return self._config['tier3_accuracy']['tier3b_evolution'].get('coverage_weight', 0.5)
    
    @property
    def tier3b_required_days(self) -> int:
        return self._config['tier3_accuracy']['tier3b_evolution'].get('required_days', 30)
    
    @property
    def flat_period_days(self) -> int:
        return self._config['reward_schedule'].get('flat_period_days', 29)
    
    @property
    def flat_reward(self) -> float:
        return self._config['reward_schedule'].get('flat_reward', 0.1)
    
    def get_config_snapshot(self) -> Dict[str, Any]:
        return self._config.copy()