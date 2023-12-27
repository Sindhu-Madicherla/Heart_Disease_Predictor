from HeartDisease.config.configuration import ConfigurationManager
from HeartDisease.components.model_evaluation import ModelEvaluation
from HeartDisease import logger

STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
            test_dataloader = model_evaluation_config.prepare_data()
            model_evaluation_config.log_into_mlflow(dataloader = test_dataloader)
        except Exception as e:
            raise e


    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e