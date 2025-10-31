from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase
from packages.jobs.celery_app import celery_app
from packages.jobs.base.base_task import BaseDataPipelineTask
from packages.jobs.base.task_models import BaseTaskContext
from packages.storage.repositories import get_connection_params, ClientFactory, get_memgraph_connection_string
from packages.storage.repositories.structural_pattern_repository import StructuralPatternRepository
from packages.storage.repositories.alerts_repository import AlertsRepository
from packages.storage.repositories.alert_cluster_repository import AlertClusterRepository
from packages.storage.repositories.alert_graph_repository import AlertGraphRepository


class SyncGraphTask(BaseDataPipelineTask):
    
    def execute_task(self, context: BaseTaskContext):
        logger.info(f"Starting Graph Sync for {context.network}")
        
        connection_params = get_connection_params(context.network)
        memgraph_connection_params = get_memgraph_connection_string(context.network)
        
        client_factory = ClientFactory(connection_params)
        with client_factory.client_context() as client:
            driver = GraphDatabase.driver(
                memgraph_connection_params['uri'],
                auth=(memgraph_connection_params['user'], memgraph_connection_params['password']) if memgraph_connection_params['user'] else None
            )
            
            try:
                pattern_repository = StructuralPatternRepository(client)
                alerts_repository = AlertsRepository(client)
                clusters_repository = AlertClusterRepository(client)
                graph_repository = AlertGraphRepository(driver)
                
                # Get snapshot data
                patterns = pattern_repository.get_deduplicated_patterns(
                    window_days=context.window_days,
                    processing_date=context.processing_date
                )
                
                if not patterns:
                    logger.warning(f"No patterns found for window_days={context.window_days}, processing_date={context.processing_date}")
                else:
                    logger.info(f"Retrieved {len(patterns)} patterns from snapshot")
                
                alerts = alerts_repository.get_all_alerts(
                    window_days=context.window_days,
                    processing_date=context.processing_date
                )
                
                if not alerts:
                    logger.warning(f"No alerts found for window_days={context.window_days}, processing_date={context.processing_date}")
                else:
                    logger.info(f"Retrieved {len(alerts)} alerts from snapshot")
                
                clusters = clusters_repository.get_all_clusters(
                    window_days=context.window_days,
                    processing_date=context.processing_date
                )
                
                if not clusters:
                    logger.warning(f"No clusters found for window_days={context.window_days}, processing_date={context.processing_date}")
                else:
                    logger.info(f"Retrieved {len(clusters)} clusters from snapshot")
                
                # Sync patterns
                if not patterns:
                    logger.warning("No patterns to sync, skipping pattern sync")
                    pattern_stats = {
                        'upserted': 0,
                        'removed': 0,
                        'total_in_graph': 0
                    }
                else:
                    pattern_stats = graph_repository.sync_snapshot(patterns, context.window_days, context.processing_date)
                
                # Sync alerts and clusters
                if not alerts and not clusters:
                    logger.warning("No alerts or clusters to sync, skipping alert/cluster sync")
                    alert_cluster_stats = {
                        'upserted_alerts': 0,
                        'upserted_clusters': 0,
                        'removed_alerts': 0,
                        'removed_clusters': 0,
                        'total_alerts_in_graph': 0,
                        'total_clusters_in_graph': 0
                    }
                else:
                    alert_cluster_stats = graph_repository.sync_alerts_and_clusters(
                        alerts if alerts else [],
                        clusters if clusters else [],
                        context.window_days,
                        context.processing_date
                    )
                
                # Get final graph stats
                graph_stats = graph_repository.get_graph_stats()
                logger.info(f"Graph stats after sync: {graph_stats}")
                
                # Build result statistics
                stats = {
                    'patterns_upserted': pattern_stats['upserted'],
                    'patterns_removed': pattern_stats['removed'],
                    'alerts_upserted': alert_cluster_stats['upserted_alerts'],
                    'alerts_removed': alert_cluster_stats['removed_alerts'],
                    'clusters_upserted': alert_cluster_stats['upserted_clusters'],
                    'clusters_removed': alert_cluster_stats['removed_clusters'],
                    'total_patterns_in_graph': pattern_stats['total_in_graph'],
                    'total_alerts_in_graph': alert_cluster_stats['total_alerts_in_graph'],
                    'total_clusters_in_graph': alert_cluster_stats['total_clusters_in_graph']
                }
                
                logger.success(f"Graph sync completed: {stats}")
                return stats
                
            finally:
                driver.close()


@celery_app.task(bind=True, base=SyncGraphTask)
def graph_sync_task(
    self,
    network: str,
    window_days: int = 7,
    processing_date: str = None
):
    context = BaseTaskContext(
        network=network,
        window_days=window_days,
        processing_date=processing_date
    )
    
    return self.run(context)
