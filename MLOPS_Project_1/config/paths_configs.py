import os

# ===== Base Project Directory =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

############### DATA INGESTION ###############
RAW_DIR = os.path.join(BASE_DIR, 'artifacts', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

RAW_FILE_PATH = os.path.join(RAW_DIR, 'raw.csv')
TRAIN_FILE_PATH = os.path.join(RAW_DIR, 'train.csv')
TEST_FILE_PATH = os.path.join(RAW_DIR, 'test.csv')

CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')


############### DATA PROCESSING ###############
PROCESSED_DIR = os.path.join(BASE_DIR, 'artifacts', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_train.csv')
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_test.csv')


############### MODEL TRAINING ###############
MODEL_DIR = os.path.join(BASE_DIR, 'artifacts', 'models')
os.makedirs(MODEL_DIR , exist_ok=True)

