import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig


# This class is responsible for setting up callbacks for a deep learning model training process.
# Callbacks like TensorBoard and ModelCheckpoint are used to monitor and save model checkpoints.

class PrepareCallback:

    # The constructor takes a config object, which contains configurations such as log directories and checkpoint file paths.
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    # Property to create a TensorBoard callback.
    # TensorBoard helps in visualizing metrics like loss, accuracy, etc., during training.
    @property
    def _create_tb_callbacks(self):
        # Create a timestamp for uniquely naming log directories for each run.
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        
        # Define the path where the TensorBoard logs will be saved.
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,  # Root directory for TensorBoard logs.
            f"tb_logs_at_{timestamp}",            # Sub-directory with the current timestamp.
        )
        
        # Return a TensorBoard callback, which will write logs to the specified directory.
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    # Property to create a ModelCheckpoint callback.
    # ModelCheckpoint saves the model during training, only saving the best version of the model (based on validation metrics).
    @property
    def _create_ckpt_callbacks(self):
        # Ensure that the filepath is passed as a string
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),  # Convert Path to string
            save_best_only=True
        )

    # Method to get both the TensorBoard and ModelCheckpoint callbacks.
    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,  # TensorBoard callback.
            self._create_ckpt_callbacks # ModelCheckpoint callback.
        ]


### not  needed to be update in pipeline as this is part of model training, i.e used when model training is happening