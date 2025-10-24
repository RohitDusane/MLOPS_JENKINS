import os
import sys
from pathlib import Path

project_name = 'MLOPS_Project_1'

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/src/__init__.py",
    f"{project_name}/src/data_ingestion.py",  
    f"{project_name}/src/data_transformation.py",
    f"{project_name}/src/model_trainer.py",
    f"{project_name}/src/logger.py",
    f"{project_name}/src/exception.py",
    #f"{project_name}/configuration/__init__.py",
    # f"{project_name}/constants/__init__.py",
    # f"{project_name}/constants/training_pipeline.py",
    # f"{project_name}/entity/__init__.py",
    # f"{project_name}/entity/config_entity.py",
    # f"{project_name}/entity/artifact_entity.py",
    # f"{project_name}/exception/__init__.py",
    # f"{project_name}/exception/exception.py",
    # f"{project_name}/logger/__init__.py",
    # f"{project_name}/logger/log.py",
    # f"{project_name}/pipeline/__init__.py",
    # f"{project_name}/pipeline/training_pipeline.py",
    # f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/artifacts/__init__.py",
    f"{project_name}/static/__init__.py",
    f"{project_name}/templates/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/notebooks/__init__.py",
    f"{project_name}/notebooks/EDA.ipynb",
    #"notebooks/1.data_collection.py",
    # "notebooks/2.EDA.py",
    #"notebooks/3.model training_prediction.py"

    #f"{project_name}/utils/main_utils.py",
    "app.py",
    "requirements.txt",
    # "Dockerfile",
    # ".dockerignore",
    #"demo.py",
    # "setup.py",
    #"config/model.yaml",
    #"config/schema.yaml",
    ".gitignore",
    #"templates/__init__.py",
    #"static/__init__.py",
    
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass
    else:
        print(f"file is already present at: {filepath}")     