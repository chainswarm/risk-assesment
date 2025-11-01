from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from datetime import datetime
from typing import List

from .models import SubmitterSubmission, BatchSubmission, ValidationScore
from packages.storage import get_connection_params, ClientFactory

router = APIRouter(prefix="/api/v1", tags=["assessment"])


@router.post("/submissions")
async def submit_scores(
    batch: BatchSubmission,
    network: str = Query(..., description="Network identifier")
):
    if not batch.submissions:
        raise HTTPException(status_code=400, detail="No submissions provided")
    
    submitter_id = batch.submissions[0].submitter_id
    processing_date = batch.submissions[0].processing_date
    
    logger.info(f"Receiving submission from miner {submitter_id} for {network} on {processing_date}")
    
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        data = []
        for sub in batch.submissions:
            data.append({
                'submission_id': sub.submission_id,
                'submitter_id': sub.submitter_id,
                'processing_date': sub.processing_date,
                'window_days': sub.window_days,
                'alert_id': sub.alert_id,
                'score': sub.score,
                'model_version': sub.model_version,
                'model_github_url': sub.model_github_url,
                'submission_timestamp': datetime.utcnow(),
                'score_metadata': sub.score_metadata or ''
            })
        
        client.insert('submissions', data, column_names=list(data[0].keys()))
        
        logger.info(f"Stored {len(data)} submissions from miner {submitter_id}")
    
    return {
        "status": "success",
        "submitter_id": submitter_id,
        "network": network,
        "processing_date": processing_date,
        "submissions_count": len(batch.submissions)
    }


@router.get("/submissions/{submitter_id}/latest")
async def get_latest_submission(
    submitter_id: str,
    network: str = Query(..., description="Network identifier")
):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        query = """
            SELECT
                processing_date,
                window_days,
                COUNT(*) as submission_count,
                model_version,
                model_github_url,
                MAX(submission_timestamp) as latest_timestamp
            FROM submissions
            WHERE submitter_id = {submitter_id:String}
            GROUP BY processing_date, window_days, model_version, model_github_url
            ORDER BY processing_date DESC, latest_timestamp DESC
            LIMIT 1
        """
        
        result = client.query(query, parameters={
            'submitter_id': submitter_id
        })
        
        if not result.result_rows:
            raise HTTPException(
                status_code=404,
                detail=f"No submissions found for miner {submitter_id} on network {network}"
            )
        
        row = result.result_rows[0]
        return {
            "submitter_id": submitter_id,
            "network": network,
            "processing_date": str(row[0]),
            "window_days": row[1],
            "submission_count": row[2],
            "model_version": row[3],
            "model_github_url": row[4],
            "latest_timestamp": row[5]
        }


@router.get("/scores/{submitter_id}/latest")
async def get_latest_validation_score(
    submitter_id: str,
    network: str = Query(..., description="Network identifier")
):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        query = """
            SELECT
                processing_date,
                window_days,
                tier1_integrity_score,
                tier2_behavior_score,
                tier3_gt_score,
                tier3_evolution_score,
                final_score,
                validation_status,
                validated_at
            FROM assessment_results
            WHERE submitter_id = {submitter_id:String}
            ORDER BY processing_date DESC, validated_at DESC
            LIMIT 1
        """
        
        result = client.query(query, parameters={
            'submitter_id': submitter_id
        })
        
        if not result.result_rows:
            raise HTTPException(
                status_code=404,
                detail=f"No validation results found for miner {submitter_id} on network {network}"
            )
        
        row = result.result_rows[0]
        return {
            "submitter_id": submitter_id,
            "network": network,
            "processing_date": str(row[0]),
            "window_days": row[1],
            "tier1_integrity_score": float(row[2]) if row[2] is not None else None,
            "tier2_behavior_score": float(row[3]) if row[3] is not None else None,
            "tier3_gt_score": float(row[4]) if row[4] is not None else None,
            "tier3_evolution_score": float(row[5]) if row[5] is not None else None,
            "final_score": float(row[6]),
            "validation_status": row[7],
            "validated_at": row[8]
        }


@router.get("/scores/rankings")
async def get_miner_rankings(
    network: str = Query(..., description="Network identifier"),
    processing_date: str = Query(..., description="Processing date (YYYY-MM-DD)"),
    window_days: int = Query(195, description="Window size in days"),
    limit: int = Query(100, description="Maximum number of results")
):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        query = """
            SELECT
                submitter_id,
                final_score,
                tier1_integrity_score,
                tier2_behavior_score,
                tier3_gt_score,
                tier3_evolution_score,
                validation_status
            FROM assessment_results
            WHERE processing_date = {processing_date:Date}
              AND window_days = {window_days:UInt32}
              AND validation_status = 'completed'
            ORDER BY final_score DESC
            LIMIT {limit:UInt32}
        """
        
        result = client.query(query, parameters={
            'processing_date': processing_date,
            'window_days': window_days,
            'limit': limit
        })
        
        if not result.result_rows:
            return []
        
        rankings = []
        for idx, row in enumerate(result.result_rows, 1):
            rankings.append({
                "rank": idx,
                "submitter_id": row[0],
                "final_score": float(row[1]),
                "tier1_integrity_score": float(row[2]) if row[2] is not None else None,
                "tier2_behavior_score": float(row[3]) if row[3] is not None else None,
                "tier3_gt_score": float(row[4]) if row[4] is not None else None,
                "tier3_evolution_score": float(row[5]) if row[5] is not None else None,
                "validation_status": row[6]
            })
        
        return rankings


@router.get("/submitters/list")
async def list_submitters(
    network: str = Query(..., description="Network identifier")
):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        query = """
            SELECT
                submitter_id,
                MAX(processing_date) as latest_date,
                argMax(final_score, processing_date) as latest_score,
                argMax(validation_status, processing_date) as latest_status,
                argMax(validated_at, processing_date) as latest_validated_at
            FROM assessment_results
            GROUP BY submitter_id
            ORDER BY latest_score DESC
        """
        
        result = client.query(query)
        
        if not result.result_rows:
            return []
        
        miners = []
        for row in result.result_rows:
            miners.append({
                "submitter_id": row[0],
                "latest_processing_date": str(row[1]),
                "latest_score": float(row[2]),
                "validation_status": row[3],
                "validated_at": row[4]
            })
        
        return miners