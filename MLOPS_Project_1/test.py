import joblib
import numpy as np
from config.paths_configs import MODEL_OUTPUT_PATH

loaded_model = joblib.load(MODEL_OUTPUT_PATH)
# loaded_model = joblib.load("XGBoost_best_model.pkl")
features = np.array([[90,1,150.0,12,20,2,3,2,2,3]])  # same as your form

prediction = loaded_model.predict(features)
print(prediction)


import pandas as pd

y = pd.read_csv('your_training_labels.csv')
print(y.value_counts(normalize=True))
