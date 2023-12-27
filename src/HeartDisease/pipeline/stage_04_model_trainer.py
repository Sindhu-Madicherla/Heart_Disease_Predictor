from HeartDisease.config.configuration import ConfigurationManager
from HeartDisease.components.model_trainer import ModelTrainer
from HeartDisease import logger

STAGE_NAME = "Model Training"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            train_dataloader = model_trainer_config.prepare_data()
            model_trainer_config.train(dataloader = train_dataloader)
        except Exception as e:
            raise e


    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
