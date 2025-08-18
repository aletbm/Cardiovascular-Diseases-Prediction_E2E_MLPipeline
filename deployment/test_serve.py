import requests

API_URL = "https://cardiovascular-diseases-api-761922006747.us-east1.run.app/predict"
# API_URL = "http://localhost:8080/predict"

sample_input = {
    'Age': 60, 
    'Sex': 'M', 
    'ChestPainType': 'ASY', 
    'RestingBP': 160, 
    'Cholesterol': 0, 
    'FastingBS': 1, 
    'RestingECG': 'Normal', 
    'MaxHR': 149, 
    'ExerciseAngina': 'N', 
    'Oldpeak': 0.4, 
    'ST_Slope': 'Flat', 
    #'HeartDisease': 1
 }


sample_input = {"features": sample_input}

response = requests.post(API_URL, json=sample_input)

print(f"Status code: {response.status_code}")
print("Prediction:", response.json())
