from HeartDisease import logger
from HeartDisease.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from HeartDisease.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from HeartDisease.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from HeartDisease.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from HeartDisease.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline


STAGE_NAME = "Data Ingestion"
try:
   logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
     
STAGE_NAME = "Data Validation"
try:
   logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
          
STAGE_NAME = "Data Transformation"
try:
   logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
   obj = DataTransformationTrainingPipeline()
   obj.main()
   logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "Model Training"
try:
   logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
   obj = ModelTrainerTrainingPipeline()
   obj.main()
   logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "Model Evaluation"
try:
   logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
   obj = ModelEvaluationPipeline()
   obj.main()
   logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e