import os
from pathlib import Path
import logging

# Configure logging to display info messages with a timestamp
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Name of the project for which files and folders are being created
project_name = "cnnClassifier"

# List of files to be created, including directory structure
list_of_files = [
    ".github/workflows/.gitkeep",  # GitHub workflow directory placeholder
    f"src/{project_name}/__init__.py",  # Initialization file for the project package
    f"src/{project_name}/components/__init__.py",  # Initialization file for components sub-package
    f"src/{project_name}/utils/__init__.py",  # Initialization file for utils sub-package
    f"src/{project_name}/config/__init__.py",  # Initialization file for config sub-package
    f"src/{project_name}/config/configuration.py",  # Configuration script file
    f"src/{project_name}/pipeline/__init__.py",  # Initialization file for pipeline sub-package
    f"src/{project_name}/entity/__init__.py",  # Initialization file for entity sub-package
    f"src/{project_name}/constants/__init__.py",  # Initialization file for constants sub-package
    "config/config.yaml",  # YAML configuration file
    "dvc.yaml",  # Data version control (DVC) pipeline file
    "params.yaml",  # Parameters file for model training or other operations
    "requirements.txt",  # Python dependencies file
    "setup.py",  # Setup script for package installation
    "research/trials.ipynb",  # Jupyter notebook for experiments or trials
    "templates/index.html"  # HTML template file
]

# Iterate over the list of files to create the necessary directories and files
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert to a Path object for OS-independent file handling
    filedir, filename = os.path.split(filepath)  # Split the path into directory and filename

    # Create the directory if it does not exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)  # Make the directories recursively if they do not exist
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create an empty file if it does not exist or if it is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")  # Log if the file already exists
