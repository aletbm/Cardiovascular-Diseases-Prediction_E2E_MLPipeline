from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deployment.serve import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
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
        #'HeartDisease': 1
    }

    sample_input = {"features": sample_input}

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200
    assert "prediction" in response.json()
