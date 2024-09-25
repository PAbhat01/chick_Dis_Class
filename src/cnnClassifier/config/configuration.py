from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,PrepareBaseModelConfig)
from cnnClassifier.entity.config_entity  import PrepareCallbacksConfig
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath= CONFIG_FILE_PATH,
        params_filepath= PARAMS_FILE_PATH
    ):
       self.config = read_yaml(config_filepath) # returned as box type so artifacts_root can be accessed using .
       self.params = read_yaml(params_filepath)
       create_directories([self.config.artifacts_root])

    # it will return a an entty i.e custom return that is defined above as dataclasss { say object}
    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config = self.config.data_ingestion  # get data ingestion from config.yaml file

        create_directories([config.root_dir])     # create the root directory mentioned in yam.config

        data_ingestion_config = DataIngestionConfig (         # create an entity {object} from the provided configuration i.e config.yaml
            root_dir= config.root_dir,              # keep root_dir as mentioned in config.yaml, similarly others
            source_url= config.source_URL,
            local_data_file= config.local_data_file,
            unzip_dir= config.unzip_dir
        )
        return data_ingestion_config  
    
    # for model tranining
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config