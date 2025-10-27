# ========== DATA PREPROCESSING ==========
"""
    Step 1 - Create load_data() to read csv files
    Step 2 - Update config.path_config.py - for data preprocessing
    Step 3 - Update config.yaml - List all columns, thresholds, vearibales required for prepreocessing (skenwess, no_of)Features)
    Step 4 - Update requirements.txt for Imbalanced Data
    Step 5 - Create src.data_preprocessing.py pipeline 
"""


import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from config.paths_configs import *
from utils.common_functions import read_yaml, load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import joblib



class DataPreprocessing():
    def __init__(self, train_path, test_path, processed_dir, config_path):
        
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        try:
            # drop unwanted columns from df
            logging.info('Dropping Columns')
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True, errors='ignore')
            logging.info(f'Booking Status Counts : \n{df["booking_status"].value_counts()}')
            logging.info('Dropping Duplicates')
            df.drop_duplicates(inplace=True)
            logging.info(f'Data Shape After Drop_Duplicates : {df.shape}')
            # Map booking_status to 1/0
            df['booking_status'] = df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})
            logging.info(f'Booking Status Counts After Drop Duplicates: \n{df["booking_status"].value_counts()}')

            # derive categorical & numerical columns
            cat_cols = self.config['data_preprocessing']['categorical_columns']
            num_cols = self.config['data_preprocessing']['numerical_columns']

            # Perform VIF - but our data is not affected hence no need to preprocess VIF
            # Label Encodig
            logging.info('Applying Label Encoder to categorical columns')
            # le = LabelEncoder()
            # mappings = {}

            # for col in cat_cols:
            #     df[col] = le.fit_transform(df[col].astype(str))
            #     mappings[col] = {label:code for label,code in zip(le.classes_, le.transform(le.classes_))}
            # logging.info('Label Encoded Mappings are: ')
            # for col, mapping in mappings.items():
            #     logging.info(f"{col} : {mapping}")

            # joblib.dump(le, f"{MODEL_OUTPUT_PATH}/label_encoder_{col}.pkl")


            # # perfrom Skewess
            # logging.info('Data Skewness Handling')
            # skew_threshold = self.config['data_preprocessing']['skewness_threshold']
            # skewness = df[num_cols].apply(lambda x:x.skew())
            # for column in skewness[skewness>skew_threshold].index:
            #     df[column] = np.log1p(df[column])
            # logging.info(f"Skewed columns (>{skew_threshold}): {list(skewness[skewness > skew_threshold].index)}")
            # logging.info(f"Applied log1p to {column} for {sum(df[column] > 0)} rows")

            # ----- Label Encoding -----
            label_encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                # Save encoder for Flask reuse
                joblib.dump(le, os.path.join(PREPROCESSING_ARTIFACTS_DIR, f"label_encoder_{col}.pkl"))
            logging.info("Label encoders saved for categorical columns.")

            # ----- Skewness Handling -----
            logging.info('Handling Data Skewness')
            skew_threshold = self.config['data_preprocessing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x: x.skew())
            skewed_columns = skewness[skewness > skew_threshold].index.tolist()

            for col in skewed_columns:
                df[col] = np.log1p(df[col])

            # Save skewed columns list
            joblib.dump(skewed_columns, os.path.join(PREPROCESSING_ARTIFACTS_DIR, "skewed_columns.pkl"))
            logging.info(f"Skewed columns handled and saved: {skewed_columns}")

            return df
        except Exception as e:
            logging.error(f"Error during data pre-processing: {e}")
            raise CustomException("Failed to pre-process data.", e)
    
    def balance_data(self,df):
        try:
            logging.info('Handling Imbalanced Data')
            
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            # Log original distribution
            logging.info(f'Data Shape BEFORE SMOTE: {X.shape[0]}')
            logging.info(f'Class distribution BEFORE SMOTE:\n{y.value_counts()}')

            smote = SMOTE(random_state=24)
            X_resampled, y_resampled = smote.fit_resample(X,y)
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled

            # Log new distribution
            logging.info(f'Data Shape AFTER SMOTE: {X_resampled.shape[0]}')
            logging.info(f'Class distribution AFTER SMOTE:\n{y_resampled.value_counts()}')
            logging.info('Data balancing successful')

            return balanced_df
        
        except Exception as e:
            logging.error(f"Error during data balancing step: {e}")
            raise CustomException("Failed to Balanced data.", e)
        
    def select_features(self, df):
        try:
            logging.info('Feature Selection Step')

            # Split X and y
            X = df.drop(columns='booking_status')
            y = df['booking_status']

            # Train RandomForest to get feature importances
            model = RandomForestClassifier(random_state=24)
            model.fit(X, y)

            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importances': model.feature_importances_
            })

            # Sort features by importance
            top_features_importances_df = feature_importance_df.sort_values(
                by='Importances', ascending=False
            )

            # Select top N features dynamically
            num_features_to_select = self.config['data_preprocessing']['no_of_features']
            top_features = top_features_importances_df['Feature'].head(num_features_to_select)
            logging.info(f'Features selected : \n {top_features.tolist()}')

            # Save top features list for later use in app
            joblib.dump(
                top_features.tolist(), 
                os.path.join(PREPROCESSING_ARTIFACTS_DIR, "top_features.pkl")
            )

            # Return dataframe with top features + target
            top_features_df = df[top_features.tolist() + ['booking_status']]
            logging.info('Feature selection completed successfully')

            return top_features_df

        except Exception as e:
            logging.error(f"Error during feature selection step: {e}")
            raise CustomException("Failed to Select Features", e)

    

    def run_preprocessing(self):
        try:
            # logging.info("Starting full preprocessing pipeline...")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            # train_df = self.balance_data(train_df)
            # test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            train_df.to_csv(PROCESSED_TRAIN_DATA_PATH, index=False)
            test_df.to_csv(PROCESSED_TEST_DATA_PATH, index=False)
            logging.info("Data preprocessing completed successfully.")

        except Exception as e:
            logging.error(f"Error during saving data step: {e}")
            raise CustomException("Failed to Save Data", e)
        finally:
            logging.info("Data Preprocessing pipeline ended.")

        
# Testing
if __name__ =="__main__":
    logging.info('Data Preprocessing pipeline starting...')
    data_processor = DataPreprocessing(train_path=TRAIN_FILE_PATH,
                                  test_path=TEST_FILE_PATH,
                                  processed_dir=PROCESSED_DIR,
                                  config_path=CONFIG_PATH)
    data_processor.run_preprocessing()