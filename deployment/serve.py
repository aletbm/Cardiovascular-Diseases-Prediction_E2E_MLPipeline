from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import cloudpickle
import os
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

with open(fs_vif_path, 'r') as f:
    selected_features = json.load(f)

with open(ss_path, "rb") as f:
    ss = cloudpickle.load(f)

with open(fs_path, "rb") as f:
    fs = cloudpickle.load(f)


app = FastAPI()


class InputData(BaseModel):
    features: dict


@app.get("/")
def read_root():
    return {"message": "API succesfully working"}


def transform_data(X, continuos_f, categorical_f):
    X["Sex"] = X["Sex"].map({'M':1, 'F':0})
    X["ExerciseAngina"] = X["ExerciseAngina"].map({'N':0, 'Y':1})

    X[ohe.get_feature_names_out()] = ohe.transform(X[categorical_f]).toarray().astype('int8')
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
