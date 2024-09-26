from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.training import Training
from cnnClassifier import logger



STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()  # Load configurations.
        prepare_callbacks_config = config.get_prepare_callback_config()  # Get callback configurations.
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)  # Prepare the callback object.
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()  # Get the list of callbacks (TensorBoard and ModelCheckpoint).

        training_config = config.get_training_config()  # Get training configurations.
        training = Training(config=training_config)  # Initialize the training class with configurations.
        training.get_base_model()  # Load the pre-trained base model.
        training.train_valid_generator()  # Prepare the training and validation data generators.
        training.train(
            callback_list=callback_list  # Start training the model with the callback list.
        )


if __name__ == '__main__':

    try:
        logger.info(f"*************************")
        logger.info(f">>>>>>>>>>>>>>  stage {STAGE_NAME} starte <<<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>> stage {STAGE_NAME} completed   <<<<<<<<<<<\n\nx===============x")
    except Exception as e :
        logger.exception(e)
        raise e