# ========== MODEL TRAINING ==========
"""
    # Step 1 - Create load_data() to read csv files
    Step 1 - Update config.path_config.py - for model training
    Step 2 - Update config.yaml - List all thresholds, variables required for model training
    Step 3 - Update requirements.txt for all models
    Step 4 - Create src.model_training.py pipeline 
"""


import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import logging
from src.exception import CustomException
from config.paths_configs import *
from utils.common_functions import read_yaml, load_data

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # optional but great for performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix, log_loss
from scipy.stats import randint,uniform

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from scipy.special import expit


class ModelTraining:
    def __init__(self, train_path, test_path, model_dir, config):
        
        self.config = config['model_training']
        self.train_path = train_path
        self.test_path = test_path
        self.model_dir = model_dir
        self.model_cm_path = os.path.join(self.model_dir, "images")
        self.baseline_cm_path = os.path.join(self.model_cm_path, "baseline")
        self.tuned_cm_path = os.path.join(self.model_cm_path, "tuned")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.model_cm_path):
            os.makedirs(self.model_cm_path)
        if not os.path.exists(self.baseline_cm_path):    
            os.makedirs(self.baseline_cm_path)
        if not os.path.exists(self.tuned_cm_path):    
            os.makedirs(self.tuned_cm_path)

        # Define baseline models
        self.baseline_models = {
            #'LogisticRegression': LogisticRegression(max_iter=1000,solver='saga'),
            #'RandomForest': RandomForestClassifier(random_state=24),
            # 'SVC': SVC(probability=True,random_state=24),
            'GradientBoosting': GradientBoostingClassifier(random_state=24),
            'AdaBoost': AdaBoostClassifier(random_state=24),
            'DecisionTree': DecisionTreeClassifier(random_state=24),
            'KNN': KNeighborsClassifier(),
            # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=24),
            'LGBM': LGBMClassifier(verbose=1, random_state=24),
        }


        # Hyperparameter grids for tuning top 3 models
        self.param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.01, 0.1]
            },
            'LGBM': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 100],
                'learning_rate': uniform(0.01, 0.2),
                'boosting_type' : ['gbdt', 'dart'],
                "min_data_in_leaf": [10, 20, 50]
            },
            "LogisticRegression": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            },
            "GradientBoosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "SVC": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            },
            "KNN": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2]
            },
        }
        
    def load_and_split_data(self):
        try:
            # Load the data from `Artifacts\Processed\`
            logging.info(f"Loading TRAIN data from {self.train_path}")
            train_df = load_data(self.train_path)

            logging.info(f'Loading TEST data from {self.test_path}')
            test_df = load_data(self.test_path)

            # Split data into X_train, y_train, X_test, y_test
            X_train = train_df.drop('booking_status', axis=1)
            y_train = train_df['booking_status']
            
            X_test = test_df.drop('booking_status', axis=1)
            y_test = test_df['booking_status']
            logging.info("Data splitted sucefully for Model Training")
            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            logging.error(f"Error during load-split data step: {e}")
            raise CustomException("Failed to load-split data.", e)
        
        
    def evaluate_model(self, model, X_test, y_test, stage="baseline"):
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = None

            # Get probabilities for ROC AUC / log_loss
            if hasattr(model, "predict_proba"):
                y_prob_full = model.predict_proba(X_test)  # shape (n_samples, 2)
                y_prob = y_prob_full[:, 1]  # probability of class 1 = Cancelled
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
                y_prob = expit(y_prob)

            # Metrics (positive class = 1 = Cancelled)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, pos_label=1),
                'precision': precision_score(y_test, y_pred, pos_label=1),
                'recall': recall_score(y_test, y_pred, pos_label=1)
            }

            # ROC AUC
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                metrics['log_loss'] = log_loss(y_test, y_prob_full if hasattr(model, "predict_proba") else y_prob)

                
            # --- Save confusion matrix ---
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cancelled", "Not Cancelled"], yticklabels=["Cancelled", "Not Cancelled"], linewidths=0.5, linecolor='gray')
            plt.title(f"{model.__class__.__name__} Confusion Matrix ({stage})")
            plt.ylabel("Actual", fontsize=12)
            plt.xlabel("Predicted", fontsize=12)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.tight_layout()

            # Save to baseline or tuned subfolder
            if stage == "baseline":
                save_path = os.path.join(self.baseline_cm_path, f"{model.__class__.__name__}_cm.png")
            else:
                save_path = os.path.join(self.tuned_cm_path, f"{model.__class__.__name__}_cm.png")

            plt.savefig(save_path)
            plt.close()
            # logging.info(f"Saved confusion matrix for {model.__class__.__name__} ({stage}) at {save_path}")

            return metrics

        except Exception as e:
            logging.error(f"Error during model evaluation step: {e}")
            raise CustomException("Failed to evaluate model.", e)
        
    # def log_experiment(self, model_name, model, metrics, params=None, stage='baseline'):
    #     uri = self.config.get('set_tracking_uri')
    #     mlflow.set_tracking_uri(uri=uri)
    #     experiment_name = self.config.get('experiment_name', 'Model_Training')
    #     mlflow.set_experiment(experiment_name)
          
    # # Each tuning stage is a new run within the same experiment
    #     run_name = f"{model_name}_{stage}"

    #     with mlflow.start_run(run_name=run_name):
    #         if params:
    #             mlflow.log_params(params)
    #         else:
    #             mlflow.log_params(model.get_params())
    #         mlflow.log_metrics(metrics)
    #         mlflow.sklearn.log_model(model, f"{model_name}_{stage}_model")

    #         # Add useful tags
    #         mlflow.set_tag("stage", stage)
    #         mlflow.set_tag("model_name", model_name)
    #         mlflow.set_tag("source", "model_training_pipeline")

    #         logging.info(f"Logged {stage} model run for {model_name} to MLflow.")

    def run_baseline_models(self):
        try:
            X_train, y_train, X_test, y_test = self.load_and_split_data()
            baseline_results = []
            
            logging.info("Starting baseline model training and evaluation...")
            
            for model_name, model in self.baseline_models.items():
                try:
                    logging.info(f"Training baseline model: {model_name}")
                    model.fit(X_train, y_train)
                    metrics = self.evaluate_model(model, X_test, y_test, stage="baseline")
                    baseline_results.append({'model_name': model_name, 'model': model, 'metrics': metrics})
                    # self.log_experiment(model_name, model, metrics, stage='baseline')
                except Exception as e:
                    logging.error(f"Error training model {model_name}: {e}")
            
            # Sort models by primary metric (ROC-AUC if available else F1)
            baseline_results.sort(key=lambda x: (x['metrics']['roc_auc'] if x['metrics']['roc_auc'] is not None else x['metrics']['f1_score']), reverse=True)
            
            # Select top 3 models
            top_n_count = self.config.get('top_n', 3)  # Default to 3 if not set
            top_n = baseline_results[:top_n_count]
            logging.info(f"Top {len(top_n)} baseline models: {[m['model_name'] for m in top_n]}")

            
            return top_n, X_train, y_train, X_test, y_test
        except Exception as e:
            logging.error(f"Error during baseline model training step: {e}")
            raise CustomException("Failed to train model (baseline).", e)
    
    def save_best_model(self, best_model_info):
        model_name = best_model_info['model_name']
        model = best_model_info['model']
        model_path = os.path.join(self.model_dir, f"{model_name}_best_model.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Best model saved at {model_path}")

        # ✅ Also save/update the global Flask model path
        joblib.dump(model, MODEL_OUTPUT_PATH)
        logging.info(f"Flask model updated at {MODEL_OUTPUT_PATH}")

    
    def hyperparameter_tuning(self, top_models, X_train, y_train, X_test, y_test):
        """
        Perform hyperparameter tuning on top N baseline models using RandomizedSearchCV.
        Selects the best model based on ROC-AUC or F1-score.

        Args:
            top_models (list): List of top baseline models with name & metrics
            X_train, y_train, X_test, y_test: Data for tuning and validation

        Returns:
            tuned_results (list): List of tuned models and their metrics
        """
        try:
            tuned_results = []
            logging.info("Starting hyperparameter tuning for top models...")

            for model_info in top_models:
                model_name = model_info['model_name']
                base_model = model_info['model']

                # Skip models with no param grid
                if model_name not in self.param_grids:
                    logging.warning(f"No hyperparameter grid found for {model_name}, skipping tuning.")
                    continue

                param_grid = self.param_grids[model_name]
                logging.info(f"Performing hyperparameter tuning for {model_name} with grid: {param_grid}")

                # Initialize RandomizedSearchCV
                randomized_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=self.config.get('n_iter', 5),
                    scoring=self.config.get('scoring', 'f1'),
                    cv=self.config.get('cv_folds', 5),
                    verbose=2,
                    n_jobs=-1,
                    random_state=24
                )

                randomized_search.fit(X_train, y_train)
                best_model = randomized_search.best_estimator_
                best_params = randomized_search.best_params_
                logging.info(f"Best parameters for {model_name}: {best_params}")

                # Evaluate tuned model
                metrics = self.evaluate_model(best_model, X_test, y_test, stage="tuned")
                tuned_results.append({
                    'model_name': model_name,
                    'model': best_model,
                    'metrics': metrics,
                    'best_params': best_params
                })

                # Log tuning experiment in MLflow
                # self.log_experiment(model_name, best_model, metrics, best_params, stage='tuned')

            # Sort by primary metric (ROC-AUC preferred, else F1)
            tuned_results.sort(key=lambda x: (x['metrics']['roc_auc'] if x['metrics']['roc_auc'] is not None else x['metrics']['f1_score']),reverse=True)

            best_model_info = tuned_results[0] if tuned_results else None

            if best_model_info:
                logging.info(f"Best tuned model: {best_model_info['model_name']} "
                                f"with metrics: {best_model_info['metrics']}")
            else:
                logging.warning("No model successfully tuned.")

            return tuned_results

        except Exception as e:
            logging.error(f"Error during hyperparameter tuning step: {e}")
            raise CustomException("Failed during hyperparameter tuning.", e)



    def run(self):
        try:
            # logging.info("Model training pipeline started.")
            
            # Step 1: Baseline models training + evaluation + logging
            top_models, X_train, y_train, X_test, y_test = self.run_baseline_models()
            
            # Step 2: Hyperparameter tuning on top 3 models
            tuned_models = self.hyperparameter_tuning(top_models, X_train, y_train, X_test, y_test)
            
            # Step 3: Save best tuned model
            if tuned_models:
                best_model = tuned_models[0]
                logging.info(f"Best tuned model: {best_model['model_name']} with metrics: {best_model['metrics']}")
                self.save_best_model(best_model)
            else:
                # Fallback: if tuning not done, save best baseline model
                logging.warning("No tuned models found, saving best baseline model instead.")
                self.save_best_model(top_models[0])
            
            logging.info("Model training pipeline completed successfully.")
        
        except Exception as e:
            logging.error(f"Error in Model Training Pipeline: {e}")
            raise CustomException("Model Training pipeline failed.", e)
        

# # Testing
# if __name__ == "__main__":
#     logging.info('Model Training pipeline starting...')
#     config = read_yaml(CONFIG_PATH)
#     model_training = ModelTraining(
#         train_path=PROCESSED_TRAIN_DATA_PATH,
#         test_path=PROCESSED_TEST_DATA_PATH,
#         model_dir=MODEL_DIR,
#         config=config
#     )
#     model_training.run()