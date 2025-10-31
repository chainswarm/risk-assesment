#!/usr/bin/env python3
import argparse
from loguru import logger
from packages.storage import ClientFactory, get_connection_params
from packages.validation import EvolutionValidator


def track_evolution(
    network: str,
    base_date: str,
    window_days: int
):
    logger.info(f"Tracking evolution for {network} from {base_date}")
    
    connection_params = get_connection_params(network)
    client_factory = ClientFactory(connection_params)
    
    validator = EvolutionValidator(client_factory)
    validator.track_evolution(base_date, window_days)
    
    logger.info("Evolution tracking complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--base-date", required=True)
    parser.add_argument("--window-days", type=int, default=195)
    
    args = parser.parse_args()
    
    track_evolution(
        args.network,
        args.base_date,
        args.window_days
    )