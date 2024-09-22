from cnnClassifier import logger
"""
 The __init__.py file in the cnnClassifier directory is executed when you import cnnClassifier. 
 This file is used to initialize the package and can also include code that makes objects"""

from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# after this run python main.py after deleting artifacts folder to check if the data is downloaded again fine
# then data ingestion pipeline is working fine


# preparing the base model

STAGE_NAME = "Prepare Base Model"
try:
        logger.info(f"**************")
        logger.info(f">>>>>. stage {STAGE_NAME} started  <<<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>  stage {STAGE_NAME} completed  <<<<<<<<< \n\n x===========x")
except Exception as e:
        logger.exception(e)
        raise e

# again delete artifacts and check if both steps of pipeline are working fine


