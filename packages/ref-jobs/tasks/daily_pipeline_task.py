from dotenv import load_dotenv
from loguru import logger
from packages.jobs.celery_app import celery_app
from packages.jobs.base.base_task import BaseDataPipelineTask
from packages.jobs.base.task_models import BaseTaskContext
from packages.jobs.tasks import InitializeAnalyzersTask
from packages.jobs.tasks.build_features_task import BuildFeaturesTask
from packages.jobs.tasks.detect_structural_patterns_task import DetectStructuralPatternsTask
from packages.jobs.tasks.detect_typologies_task import DetectTypologiesTask
from packages.jobs.tasks.graph_sync_task import SyncGraphTask
from packages.jobs.tasks.export_batch_task import ExportBatchTask


class DailyPipelineTask(BaseDataPipelineTask):
    
    def execute_task(self, context: BaseTaskContext):
        logger.info(f"Starting Daily Analytics Pipeline for {context.network}")

        logger.info("Step 1: Initialize Analyzers")
        initializer_analyzers = InitializeAnalyzersTask()
        initializer_analyzers.execute_task(context)

        logger.info("Step 2: Build Features")
        features_task = BuildFeaturesTask()
        features_task.execute_task(context)
        
        logger.info("Step 3: Detect Structural Patterns")
        structural_patterns_task = DetectStructuralPatternsTask()
        structural_patterns_task.execute_task(context)
        
        logger.info("Step 4: Detect Typologies")
        typologies_task = DetectTypologiesTask()
        typologies_task.execute_task(context)
        
        logger.info("Step 5: Sync Graph Snapshot")
        graph_sync_task = SyncGraphTask()
        graph_sync_task.execute_task(context)
        
        logger.info("Step 6: Export Batch for Miners")
        export_task = ExportBatchTask()
        export_task.run(
            network=context.network,
            processing_date=context.processing_date,
            window_days=context.window_days
        )

        logger.success(f"Daily Analytics Pipeline completed for {context.network}")

@celery_app.task(bind=True, base=DailyPipelineTask)
def daily_pipeline_task(
    self,
    network: str,
    window_days: int = 7,
    min_edge_weight: float = 100.0,
    sampling_percentage: float = 0.0,
    processing_date: str = None
):
    context = BaseTaskContext(
        network=network,
        window_days=window_days,
        processing_date=processing_date,
        min_edge_weight=min_edge_weight,
        sampling_percentage=sampling_percentage,
        batch_size=1024,

    )
    
    return self.run(context)
