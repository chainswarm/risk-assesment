from dotenv import load_dotenv

from packages.jobs.base.task_models import BaseTaskContext
from packages.jobs.celery_app import celery_app
from packages.jobs.base.base_task import BaseDataPipelineTask
from packages.storage.repositories import get_connection_params, ClientFactory, MigrateSchema
from packages import setup_logger


class InitializeAnalyzersTask(BaseDataPipelineTask):

    def execute_task(self, context: BaseTaskContext):
        service_name = f'analytics-{context.network}-initialize-analyzers'
        setup_logger(service_name)

        connection_params = get_connection_params(context.network)
        client_factory = ClientFactory(connection_params)
        with client_factory.client_context() as client:
            migrate_schema = MigrateSchema(client)
            migrate_schema.run_analyzer_migrations()


@celery_app.task(bind=True, base=InitializeAnalyzersTask)
def initialize_analyzers_task(self, network: str, window_days: int, processing_date: str):
    context = BaseTaskContext(
        network=network,
        window_days=window_days,
        processing_date=processing_date
    )

    return self.run(context)

