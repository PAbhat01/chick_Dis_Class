from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig

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