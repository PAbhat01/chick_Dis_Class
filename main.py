from cnnClassifier import logger
"""
 The __init__.py file in the cnnClassifier directory is executed when you import cnnClassifier. 
 This file is used to initialize the package and can also include code that makes objects"""

from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline



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