import joblib
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash
from config.paths_configs import MODEL_OUTPUT_PATH, PREPROCESSING_ARTIFACTS_DIR

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# ---------------- Load Model & Preprocessing Artifacts ----------------
loaded_model = joblib.load(MODEL_OUTPUT_PATH)

# Categorical columns & their label encoders
cat_cols = ['market_segment_type']  
label_encoders = {
    col: joblib.load(os.path.join(PREPROCESSING_ARTIFACTS_DIR, f"label_encoder_{col}.pkl"))
    for col in cat_cols
}

# Skewed numeric columns
skewed_columns = joblib.load(os.path.join(PREPROCESSING_ARTIFACTS_DIR, "skewed_columns.pkl"))

# Dynamically load top features
top_features = joblib.load(os.path.join(PREPROCESSING_ARTIFACTS_DIR, "top_features.pkl"))

# For dropdowns in HTML, get all unique classes for categorical columns
cat_options = {
    col: list(label_encoders[col].classes_)
    for col in cat_cols
}

# -------------------- Flask Routes --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_data = {}

    if request.method == 'POST':
        try:
            # Collect user inputs
            for feature in top_features:
                val = request.form.get(feature)
                if val is None or val.strip() == '':
                    flash(f"Missing value for {feature.replace('_', ' ').title()}")
                    return render_template('index.html', prediction=None, top_features=top_features, cat_options=cat_options)

                # Convert numeric values, leave categorical as string
                if feature in cat_cols:
                    input_data[feature] = val
                else:
                    input_data[feature] = float(val)

            df_input = pd.DataFrame([input_data])

            # Label encode categorical columns
            for col in cat_cols:
                if col in df_input.columns:
                    if df_input[col][0] not in label_encoders[col].classes_:
                        flash(f"Invalid value '{df_input[col][0]}' for {col}")
                        return render_template('index.html', prediction=None, top_features=top_features, cat_options=cat_options)
                    df_input[col] = label_encoders[col].transform(df_input[col].astype(str))

            # Apply log1p to skewed numeric columns
            for col in skewed_columns:
                if col in df_input.columns:
                    df_input[col] = np.log1p(df_input[col])

            # Reorder columns as per model training
            df_input = df_input[top_features]

            # Predict
            prediction = loaded_model.predict(df_input)[0]

        except Exception as e:
            flash(f"Error processing input: {e}")
            return render_template('index.html', prediction=None, top_features=top_features, cat_options=cat_options)

    return render_template('index.html', prediction=prediction, top_features=top_features, cat_options=cat_options)

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
