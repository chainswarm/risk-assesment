#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from packages import setup_logger
from packages.storage import ClientFactory, get_connection_params, create_database


def create_assessment_tables(network: str):
    schema_dir = Path(__file__).parent.parent / "packages" / "storage" / "schema"
    
    tables = [
        "miner_submissions",
        "feature_evolution_tracking",
        "miner_validation_results"
    ]
    
    connection_params = get_connection_params(network)
    create_database(connection_params)
    
    client_factory = ClientFactory(connection_params)
    
    with client_factory.client_context() as client:
        for table in tables:
            schema_file = schema_dir / f"{table}.sql"
            
            if not schema_file.exists():
                raise ValueError(f"Schema file not found: {schema_file}")
            
            sql = schema_file.read_text()
            client.command(sql)
            logger.info(f"Created table {table}")


def main():
    parser = argparse.ArgumentParser(description="Initialize assessment tables in ClickHouse")
    parser.add_argument('--network', type=str, required=True, help='Network identifier (ethereum, bitcoin, torus, etc.)')
    args = parser.parse_args()
    
    service_name = f'{args.network}-assessment-tables-init'
    setup_logger(service_name)
    load_dotenv()
    
    logger.info(
        "Initializing assessment tables",
        extra={
            "network": args.network,
        }
    )
    
    create_assessment_tables(args.network)
    
    logger.info("Assessment tables initialization completed")


if __name__ == "__main__":
    main()