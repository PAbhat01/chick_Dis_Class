import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

# Define the PrepareBaseModel class which is responsible for creating, updating, and saving a base model.
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Args:
        - config: An instance of the PrepareBaseModelConfig class that contains the parameters required for the model.
        """
        self.config = config  # Store the config for use in the class

    def get_base_model(self):
        """
        Create the base model using VGG16 architecture with parameters defined in the config.
        The model is saved to the specified base model path after being created.
        """
        # Create the VGG16 base model with the parameters from the config
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,  # Shape of the input image
            weights=self.config.params_weights,  # Pretrained weights to use (e.g., 'imagenet')
            include_top=self.config.params_include_top  # Whether to include the top classification layer
        )

        # Save the base model after it's created
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare a full model by adding a custom classification head on top of the base model and compiling it.
        
        Args:
        - model: The base model to which a classification head will be added.
        - classes: Number of output classes for the classification task.
        - freeze_all: Whether to freeze all layers in the base model for training.
        - freeze_till: Number of layers from the end of the model that should remain trainable (None if freezing all).
        - learning_rate: The learning rate to use for model training.

        Returns:
        - full_model: The modified full model with a custom classification head and compilation.
        """
        # Freeze layers of the base model based on the freeze_all or freeze_till parameter
        if freeze_all:
            for layer in model.layers:
                model.trainable = False  # Freeze all layers
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]: # freeze_till should be number of layers that are not to be trained
                model.trainable = False  # Freeze layers until freeze_till, keeping the rest trainable

        # Add a Flatten layer to convert 2D features into a 1D feature vector
        flatten_in = tf.keras.layers.Flatten()(model.output)

        # Add a Dense layer for classification with softmax activation for multi-class output
        prediction = tf.keras.layers.Dense(
            units=classes,  # Number of classes
            activation="softmax"  # Use softmax activation for multi-class classification
        )(flatten_in)

        # Create the full model by combining the base model with the new prediction layer
        full_model = tf.keras.models.Model(
            inputs=model.input,  # Input from the base model
            outputs=prediction  # Output from the new Dense layer
        )

        # Compile the model with SGD optimizer and categorical cross-entropy loss
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),  # Use Stochastic Gradient Descent
            loss=tf.keras.losses.CategoricalCrossentropy(),  # Loss for multi-class classification
            metrics=["accuracy"]  # Track accuracy as the performance metric
        )

        full_model.summary()  # Print a summary of the model architecture
        return full_model  # Return the complete model

    def update_base_model(self):
        """
        Update the base model by adding a classification head and freezing all layers.
        The updated model is saved to the specified path.
        """
        # Prepare the full model with the classification head and freeze all layers
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,  # Number of output classes
            freeze_all=True,  # Freeze all layers in the base model
            freeze_till=None,  # No layers left trainable
            learning_rate=self.config.params_learning_rate  # Learning rate for training
        )

        # Save the updated model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the given model to the specified path.

        Args:
        - path: The file path where the model should be saved.
        - model: The TensorFlow model to be saved.
        """
        model.save(path)  # Save the model to the path