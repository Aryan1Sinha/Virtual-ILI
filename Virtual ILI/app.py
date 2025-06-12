from flask import Flask, render_template, request
import joblib
import numpy as np

import pandas as pd
import os
from flask import send_file

app = Flask(__name__)

# Load trained models
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')

# Define input features in the correct order
FEATURES = [
    'Pipe_age',
    'Coating',
    'Operating_pressure(psi)',
    'Pressure cycles',
    'Flow rate (bbl/d)',
    'Temperature ©',
    'H2S_concentration',
    'CO2_concentration',
    'Water_cut',
    'Previous wall loss '
]

@app.route('/')
def index():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(inputs).reshape(1, -1)

        reg_output = reg_model.predict(input_array)[0]
        clf_raw = clf_model.predict(input_array)[0]

        clf_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        clf_output = clf_labels.get(clf_raw, "Unknown")

        # Save to CSV
        data = {FEATURES[i]: inputs[i] for i in range(len(FEATURES))}
        data["Predicted Wall Loss (mm)"] = reg_output
        data["Corrosion Risk Level"] = clf_output

        df = pd.DataFrame([data])
        file_path = 'predictions.csv'

        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        return render_template(
            'result.html',
            reg_result=f"Predicted Wall Loss (mm): {reg_output:.2f}",
            clf_result=f"Corrosion Risk Level: {clf_output}",
            show_download=True
        )
    except Exception as e:
        return f"Something went wrong: {e}"

# Route to download CSV
@app.route('/download')
def download():
    return send_file('predictions.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
    
    


