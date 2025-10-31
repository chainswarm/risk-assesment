from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from celery import Task
from loguru import logger

from packages.jobs.celery_app import celery_app
from packages.jobs.config import SOTConfig, IngestionConfig
from packages.jobs.tasks.validation_utils import SOTDataValidator
from packages.ingestion.sot_ingestion import SOTIngestion
from packages.storage import get_connection_params, ClientFactory


class SOTIngestionTask(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True


@celery_app.task(base=SOTIngestionTask, bind=True)
def sot_ingestion_task(
    self,
    network: str,
    window_days: int,
    processing_date: Optional[str] = None
):
    if processing_date is None:
        processing_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(
        "Starting SOT ingestion task",
        network=network,
        processing_date=processing_date,
        window_days=window_days
    )
    
    sot_config = SOTConfig.from_env()
    ingestion_config = IngestionConfig.from_env()
    download_path = Path(f"/tmp/sot_data/{network}/{processing_date}")
    
    try:
        for attempt in range(1, ingestion_config.retry_attempts + 1):
            logger.info(
                "Download attempt",
                attempt=attempt,
                max_attempts=ingestion_config.retry_attempts
            )
            
            download_success = _download_sot_data(
                network=network,
                processing_date=processing_date,
                window_days=window_days,
                download_path=download_path,
                sot_config=sot_config
            )
            
            if not download_success:
                if attempt < ingestion_config.retry_attempts:
                    logger.warning(
                        "Download failed, retrying",
                        attempt=attempt,
                        delay_seconds=ingestion_config.retry_delay
                    )
                    import time
                    time.sleep(ingestion_config.retry_delay)
                    continue
                else:
                    raise ValueError("Download failed after all retry attempts")
            
            validator = SOTDataValidator(download_path)
            
            if not validator.validate_meta_exists():
                validator.cleanup_download_dir()
                if attempt < ingestion_config.retry_attempts:
                    continue
                raise ValueError("META.json validation failed")
            
            meta = validator.load_meta()
            if not meta:
                validator.cleanup_download_dir()
                if attempt < ingestion_config.retry_attempts:
                    continue
                raise ValueError("Failed to load META.json")
            
            if not validator.validate_all_files_present(meta):
                validator.cleanup_download_dir()
                if attempt < ingestion_config.retry_attempts:
                    continue
                raise ValueError("File presence validation failed")
            
            if not validator.validate_checksums(meta):
                validator.cleanup_download_dir()
                if attempt < ingestion_config.retry_attempts:
                    continue
                raise ValueError("Checksum validation failed")
            
            logger.info("All validations passed, starting ingestion")
            break
        
        _ingest_validated_data(
            network=network,
            processing_date=processing_date,
            window_days=window_days,
            download_path=download_path
        )
        
        logger.success(
            "SOT ingestion completed",
            network=network,
            processing_date=processing_date
        )
        
        return {
            'status': 'success',
            'network': network,
            'processing_date': processing_date,
            'window_days': window_days
        }
        
    except Exception as e:
        logger.error(
            "SOT ingestion failed",
            network=network,
            processing_date=processing_date,
            error=str(e)
        )
        raise
    finally:
        if download_path.exists():
            SOTDataValidator(download_path).cleanup_download_dir()


def _download_sot_data(
    network: str,
    processing_date: str,
    window_days: int,
    download_path: Path,
    sot_config: SOTConfig
) -> bool:
    try:
        download_path.mkdir(parents=True, exist_ok=True)
        
        ingestion = SOTIngestion(
            network=network,
            sot_host=sot_config.sot_host,
            sot_port=sot_config.sot_port,
            sot_database=sot_config.sot_database
        )
        
        ingestion.download_data(
            processing_date=processing_date,
            window_days=window_days,
            download_path=str(download_path)
        )
        
        return True
    except Exception as e:
        logger.error("Download failed", error=str(e))
        return False


def _ingest_validated_data(
    network: str,
    processing_date: str,
    window_days: int,
    download_path: Path
):
    connection_params = get_connection_params(network)
    client_factory = ClientFactory(connection_params)
    
    with client_factory.client_context() as client:
        tables_to_ingest = [
            'raw_alerts',
            'raw_features',
            'raw_clusters',
            'raw_address_labels'
        ]
        
        for table_name in tables_to_ingest:
            file_path = download_path / f"{table_name}.parquet"
            
            if not file_path.exists():
                logger.warning(
                    "Table file not found, skipping",
                    table=table_name,
                    path=str(file_path)
                )
                continue
            
            logger.info("Ingesting table", table=table_name)
            
            import pandas as pd
            df = pd.read_parquet(file_path)
            
            client.insert_df(table_name, df)
            
            logger.info(
                "Table ingested",
                table=table_name,
                rows=len(df)
            )