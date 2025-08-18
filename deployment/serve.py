from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import cloudpickle
import os
from fastapi.responses import HTMLResponse
import json

RUN_ID = "565143210db14184b22a8555d1d17e98"
artifacts_path = f"./models/1/{RUN_ID}/artifacts/"

model_path = os.path.join(artifacts_path, "KNeighborsClassifier/model.pkl")
ohe_path = os.path.join(artifacts_path, "preprocessing/ohe.pkl")
fs_vif_path = os.path.join(artifacts_path, "preprocessing/fs_vif.json")
ss_path = os.path.join(artifacts_path, "preprocessing/ss.pkl")
fs_path = os.path.join(artifacts_path, "preprocessing/fs.pkl")

with open(model_path, "rb") as f:
    model = cloudpickle.load(f)

with open(ohe_path, "rb") as f:
    ohe = cloudpickle.load(f)

with open(fs_vif_path, "r") as f:
    selected_features = json.load(f)

with open(ss_path, "rb") as f:
    ss = cloudpickle.load(f)

with open(fs_path, "rb") as f:
    fs = cloudpickle.load(f)


app = FastAPI()


class InputData(BaseModel):
    features: dict


def format_python_example() -> str:
    example_code = """import requests

API_URL = "https://cardiovascular-diseases-api-761922006747.us-east1.run.app/predict"
# API_URL = "http://localhost:8080/predict"

sample_input = {
    "Age": 60,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 160,
    "Cholesterol": 0,
    "FastingBS": 1,
    "RestingECG": "Normal",
    "MaxHR": 149,
    "ExerciseAngina": "N",
    "Oldpeak": 0.4,
    "ST_Slope": "Flat"
}

request_data = {"features": sample_input}

response = requests.post(API_URL, json=request_data)

print(f"Status code: {response.status_code}")
print("Prediction:", response.json())"""
    return example_code


@app.get("/", response_class=HTMLResponse)
def api_root():
    sample_input = {
        "Age": 60,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 160,
        "Cholesterol": 0,
        "FastingBS": 1,
        "RestingECG": "Normal",
        "MaxHR": 149,
        "ExerciseAngina": "N",
        "Oldpeak": 0.4,
        "ST_Slope": "Flat",
    }

    style = "padding: 8px; border: 1px solid #ddd;"
    feature_table = f"""
    <div style="overflow-x: auto;">
        <table border="1" style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 12px; text-align: left;">Feature Name</th>
                    <th style="padding: 12px; text-align: left;">Description</th>
                    <th style="padding: 12px; text-align: left;">Typical Range</th>
                    <th style="padding: 12px; text-align: left;">Type</th>
                </tr>
            </thead>
            <tbody>
                <tr><td style={style}>Age</td><td style={style}>Age of the patient [years]</td><td style={style}>> 0</td><td style={style}>Integer</td></tr>
                <tr><td style={style}>Sex</td><td style={style}>Sex of the patient [M: Male, F: Female]</td><td style={style}>M / F</td><td style={style}>Categorical</td></tr>
                <tr><td style={style}>ChestPainType</td><td style={style}>Chest pain type</td><td style={style}>TA / ATA / NAP / ASY</td><td style={style}>Categorical</td></tr>
                <tr><td style={style}>RestingBP</td><td style={style}>Resting blood pressure [mm Hg]</td><td style={style}>~ 90–200</td><td style={style}>Integer</td></tr>
                <tr><td style={style}>Cholesterol</td><td style={style}>Serum cholesterol [mg/dl]</td><td style={style}>~ 0–600</td><td style={style}>Integer</td></tr>
                <tr><td style={style}>FastingBS</td><td style={style}>Fasting blood sugar >120 mg/dl</td><td style={style}>0 or 1</td><td style={style}>Integer</td></tr>
                <tr><td style={style}>RestingECG</td><td style={style}>Resting electrocardiogram results</td><td style={style}>Normal / ST / LVH</td><td style={style}>Categorical</td></tr>
                <tr><td style={style}>MaxHR</td><td style={style}>Maximum heart rate achieved</td><td style={style}>60–202</td><td style={style}>Integer</td></tr>
                <tr><td style={style}>ExerciseAngina</td><td style={style}>Exercise-induced angina</td><td style={style}>Y / N</td><td style={style}>Categorical</td></tr>
                <tr><td style={style}>Oldpeak</td><td style={style}>ST depression induced by exercise</td><td style={style}>>= 0</td><td style={style}>Float</td></tr>
                <tr><td style={style}>ST_Slope</td><td style={style}>Slope of the peak exercise ST segment</td><td style={style}>Up / Flat / Down</td><td style={style}>Categorical</td></tr>
                <tr><td style={style}>HeartDisease</td><td style={style}>Target variable</td><td style={style}>0 = Normal, 1 = Heart disease</td><td style={style}>Binary</td></tr>
            </tbody>
        </table>
    </div>
    """

    return f"""
<html>
<head>
    <title>Cardiovascular Diseases API Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 900px; margin: 0 auto; }}
        h1 {{ color: #2a6496; border-bottom: 1px solid #eee; }}
        .endpoint {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .code-block {{ background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .try-button {{
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 10px 15px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <h1>Cardiovascular Diseases API</h1>

    <div class="endpoint">
        <h2>API Endpoint</h2>
        <p><strong>POST</strong> <code>/predict</code></p>
        <p>Accepts patient health features and returns risk of cardiovascular disease.</p>
    </div>

    <h2>Request Format</h2>
    <p>Send a JSON payload with these features:</p>

    <div class="code-block">
        <pre>{json.dumps(sample_input, indent=4)}</pre>
    </div>

    <p>The API expects the features inside a <code>"features"</code> key:</p>
    <div class="code-block">
        <pre>{json.dumps({"features": sample_input}, indent=4)}</pre>
    </div>

    <h2>Python Example</h2>
    <div class="code-block">
        <pre>{format_python_example()}</pre>
    </div>

    <h2>cURL Example</h2>
    <div class="code-block">
        <pre>curl -X POST "https://cardiovascular-diseases-api-761922006747.us-east1.run.app/predict" \\
-H "Content-Type: application/json" \\
-d '{json.dumps({"features": sample_input}, indent=2)}'</pre>
    </div>

    <h2>Expected Response</h2>
    <div class="code-block">
        <pre>{{
  "prediction": "0|1"
}}</pre>
    </div>

    <a href="/docs" class="try-button">Try it in Interactive Docs</a>

    <h2>Feature Descriptions</h2>
    {feature_table}
</body>
</html>
"""


def transform_data(X, continuos_f, categorical_f):
    X["Sex"] = X["Sex"].map({"M": 1, "F": 0})
    X["ExerciseAngina"] = X["ExerciseAngina"].map({"N": 0, "Y": 1})

    X[ohe.get_feature_names_out()] = (
        ohe.transform(X[categorical_f]).toarray().astype("int8")
    )
    X.drop(categorical_f, axis=1, inplace=True)

    X = X[selected_features].copy()
    X[continuos_f] = ss.transform(X[continuos_f])
    X = fs.transform(X)

    return X


@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    continuos_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_f = ["ChestPainType", "RestingECG", "ST_Slope"]

    df = transform_data(df, continuos_f, categorical_f)

    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
