import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


# Class for training the model. It includes methods to load the base model, prepare data generators, and execute the training process.
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    # Method to load the base model from the updated base model path.
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    # Method to prepare data generators for training and validation.
    def train_valid_generator(self):

        # Common settings for the data generator, such as rescaling and splitting data for validation.
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        # Settings for the flow_from_directory method used by the data generator.
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Create a validation data generator.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Generate validation data from the directory.
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Check if data augmentation is enabled, and prepare the training data generator accordingly.
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        # Generate training data from the directory.
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    # Static method to save the trained model to a specified path.
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
       model.save(str(path))

    # Method to train the model with the specified callback list (TensorBoard and ModelCheckpoint).
    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model using the train and validation generators, and apply the callbacks for logging and saving checkpoints.
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # Save the trained model to the specified path.
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )