from dotenv import load_dotenv
from loguru import logger

from packages.analyzers.features.address_feature_analyzer import AddressFeatureAnalyzer
from packages.analyzers.features.feature_statistics_analyzer import FeatureStatisticsAnalyzer
from packages.jobs.base import BaseTaskContext
from packages.jobs.celery_app import celery_app
from packages.jobs.base.base_task import BaseDataPipelineTask
from packages.storage.repositories import get_connection_params, ClientFactory
from packages.storage.repositories.transfer_aggregation_repository import TransferAggregationRepository
from packages.storage.repositories.money_flows_repository import MoneyFlowsRepository
from packages.storage.repositories.feature_repository import FeatureRepository
from packages.storage.repositories.transfer_repository import TransferRepository
from packages import setup_logger
from packages.utils import calculate_time_window


class BuildFeaturesTask(BaseDataPipelineTask):

    def execute_task(self, context: BaseTaskContext):

        service_name = f'features-{context.network}-build-features'
        setup_logger(service_name)

        connection_params = get_connection_params(context.network)

        client_factory = ClientFactory(connection_params)
        with client_factory.client_context() as client:
            transfer_repository = TransferRepository(client)
            transfer_aggregation_repository = TransferAggregationRepository(client)
            feature_repository = FeatureRepository(client)
            money_flows_repository = MoneyFlowsRepository(client)

            logger.info(f"Cleaning partition for window_days={context.window_days}, processing_date={context.processing_date}")
            feature_repository.delete_partition(context.window_days, context.processing_date)

            start_timestamp, end_timestamp = calculate_time_window(context.window_days, context.processing_date)

            logger.info("Starting unified feature analysis with mandatory graph analytics")
            address_analyzer = AddressFeatureAnalyzer(
                transfer_repository=transfer_repository,
                transfer_aggregation_repository=transfer_aggregation_repository,
                money_flows_repository=money_flows_repository,
                feature_repository=feature_repository,
                window_days=context.window_days,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                network=context.network,
            )

            address_analyzer.analyze_address_features(batch_size=context.batch_size)
            logger.success("Unified feature analysis with graph analytics completed successfully")

            logger.info("Starting feature statistics analysis")
            stats_analyzer = FeatureStatisticsAnalyzer(
                feature_repository=feature_repository,
                window_days=context.window_days,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                network=context.network
            )

            stats_analyzer.analyze_feature_statistics()
            logger.success("Feature statistics computed successfully")

@celery_app.task(bind=True, base=BuildFeaturesTask)
def build_features_task(
    self,
    network: str,
    window_days: int,
    processing_date: str,
    batch_size: int = 1000
):
    context = BaseTaskContext(
        network=network,
        window_days=window_days,
        processing_date=processing_date,
        batch_size=batch_size,
    )

    return self.run(context)


