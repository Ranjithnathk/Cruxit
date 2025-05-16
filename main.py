from src.cruxit.logging import logger
from src.cruxit.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated") 
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.main()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e