
################ READING YAML FILE() ################
"""Since we are reading the yaml file multiple times we create this function for reproducibility"""
import os,sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import yaml

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError("File not found in given path")
        
        with open(file_path,'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            return config
    except Exception as e:
        logging.info('Error while reading YAML file')
        raise CustomException(e, sys)
    
def load_data(path):
    try:
        # logging.info('Loading data..........')
        logging.info(f"Loading data from: {path}")
        df = pd.read_csv(path)
        logging.info(f"Data shape: {df.shape}")
        return df         
        logging.info('Loaded data Successfully..........')
    except Exception as e:
        logging.error(f'Error loading the data {e}')
        raise CustomException('Failed to load data', e)