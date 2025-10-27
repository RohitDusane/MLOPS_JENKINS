import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from config.paths_configs import *
from utils.common_functions import read_yaml

# ========== DATA INGESTION ==========

# PERFORM THIS IMPORTANT STEP
# set GOOGLE_APPLICATION_CREDENTIALS=E:\path\to\service-account-key.json



class DataIngestion:
    def __init__(self, config):
        # Load configuration for data ingestion
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_name']
        self.train_test_ratio = self.config['train_ratio']

        # Create the RAW-DIR under artifacts if it doesn't exist
        os.makedirs(RAW_DIR, exist_ok=True)

        # Log initialization information
        # logging.info(f"Data Ingested from bucket: {self.bucket_name}, & file: {self.file_name}")

    def download_csv_from_gcp(self):
        try:
            logging.info(f"Initiating download of {self.file_name} from GCP bucket {self.bucket_name}...")
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logging.info(f"CSV file successfully downloaded to {RAW_FILE_PATH}")
        except Exception as e:
            logging.error(f"Error downloading CSV file  from GCP: {e}")
            raise CustomException("Failed to download CSV file from GCP.", e)

    def split_data(self):
        try:
            logging.info("Initiating `TRAIN-TEST` data split...")
            # Read raw data
            data = pd.read_csv(RAW_FILE_PATH)
            # Split data into training and testing sets
            train_data, test_data = train_test_split(data, train_size=self.train_test_ratio, random_state=24)

            # Save the split data
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logging.info(f"Training data saved to {TRAIN_FILE_PATH}")
            logging.info(f"Testing data saved to {TEST_FILE_PATH}")
        except Exception as e:
            logging.error(f"Error during data splitting: {e}")
            raise CustomException("Failed to split data into training and test sets.", e)

    def run(self):
        try:
            # logging.info("Data Ingestion process started...")
            self.download_csv_from_gcp()  # Download CSV from GCP
            self.split_data()  # Split the downloaded data into train and test sets
            logging.info("Data Ingestion process completed successfully.")
        except CustomException as ce:
            logging.error(f"Data Ingestion failed: {str(ce)}")
        except Exception as e:
            logging.error(f"Unexpected error occurred during data ingestion: {e}")
        finally:
            logging.info("Data Ingestion pipeline ended.")

# Testing Data Ingestion pipeline
# if __name__ == "__main__":
#     logging.info('Data Ingestion pipeline starting...')
#     config = read_yaml(CONFIG_PATH)
#     data_ingestion = DataIngestion(config)
#     data_ingestion.run()