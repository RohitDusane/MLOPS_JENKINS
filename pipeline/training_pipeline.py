import sys
from .src.logger import logging
from .src.exception import CustomException
from .config.paths_configs import *
from .utils.common_functions import read_yaml

from .src.data_ingestion import DataIngestion
from .src.data_preprocessing import DataPreprocessing
from .src.model_trainer import ModelTraining

def main():
    try:
        logging.info("========== ML Pipeline Started ==========")
        config = read_yaml(CONFIG_PATH)

        # ---------------- Data Ingestion ----------------
        logging.info("Data Ingestion pipeline starting...")
        data_ingestion = DataIngestion(config)
        data_ingestion.run()
        logging.info("Data Ingestion completed successfully.")

        # ---------------- Data Preprocessing ----------------
        logging.info("Data Preprocessing pipeline starting...")
        data_processor = DataPreprocessing(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH
        )
        data_processor.run_preprocessing()
        logging.info("Data Preprocessing completed successfully.")

        # ---------------- Model Training ----------------
        logging.info("Model Training pipeline starting...")
        model_trainer = ModelTraining(
            train_path=PROCESSED_TRAIN_DATA_PATH,
            test_path=PROCESSED_TEST_DATA_PATH,
            model_dir=MODEL_DIR,
            config=config
        )
        model_trainer.run()
        logging.info("Model Training pipeline completed successfully.")

        logging.info("========== ML Pipeline Finished ==========")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise CustomException("ML Pipeline execution failed.", e)

if __name__ == "__main__":
    main()
