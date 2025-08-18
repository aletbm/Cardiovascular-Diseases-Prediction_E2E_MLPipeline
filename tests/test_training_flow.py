import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines import training_flow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from config import DATA_FOLDER, SEED_VALUE

input_folder = "./"
data_folder = input_folder + "data/"

def detect_outliers(train, columns, tipo="leve", test=None):
    train_ = train.copy()
    test_ = test.copy()
    if tipo == "extremo":
        k = 3
    else:
        k = 1.5
    describe = train_.describe()
    for col in columns:
        Q1 = train_[col].quantile(0.25)
        Q3 = train_[col].quantile(0.75)
        IQR = Q3 - Q1
        for dataset in [train_, test_]:
            interval = ((dataset[col] < Q1 - k*IQR) | (dataset[col] > Q3 + k*IQR))
            dataset.loc[interval, col] = dataset.groupby(["ChestPainType", "RestingECG", "ST_Slope"])[col].transform('median')[interval]
    return train_, test_, describe

def clean_data(X, y):
    index = X[X["Cholesterol"] == 0].index
    X.drop(index, axis=0, inplace=True)
    y.drop(index, axis=0, inplace=True)
    return X, y

def test_encode_data():
    target = "HeartDisease"
    categorical_f = ["ChestPainType", "RestingECG", "ST_Slope"]

    df = pd.read_csv(DATA_FOLDER + "heart.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop([target], axis=1), df[target], test_size=0.25, random_state=42, stratify=df[target])
    
    X_train, X_test, _ = detect_outliers(X_train, ["RestingBP", "Cholesterol", "Oldpeak"], "leve", X_test)
    X_train, y_train = clean_data(X_train, y_train)
    act_X_train, act_X_test, _ = training_flow.encode_data(X_train.copy(), X_test.copy(), categorical_f)

    for dataset in [X_train, X_test]:
        dataset["Sex"] = dataset["Sex"].map({'M':1, 'F':0})
        dataset["ExerciseAngina"] = dataset["ExerciseAngina"].map({'N':0, 'Y':1})

    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train[categorical_f])

    for dataset in [X_train, X_test]:
        dataset[ohe.get_feature_names_out()] = ohe.transform(dataset[categorical_f]).toarray().astype('int8')
        dataset.drop(categorical_f, axis=1, inplace=True)

    assert np.allclose(np.std(X_train), np.std(act_X_train))
    assert np.allclose(np.std(X_test), np.std(act_X_test))

    assert X_train.shape == act_X_train.shape
    assert X_test.shape == act_X_test.shape


def test_detect_outliers():
    target = "HeartDisease"
    continuos_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    df = pd.read_csv(DATA_FOLDER + "heart.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop([target], axis=1), df[target], test_size=0.25, random_state=42, stratify=df[target])
    
    act_X_train, act_X_test, _ = training_flow.detect_outliers(X_train.copy(), ["RestingBP", "Cholesterol", "Oldpeak"], "leve", X_test.copy())
    X_train, X_test, _ = detect_outliers(X_train, ["RestingBP", "Cholesterol", "Oldpeak"], "leve", X_test)

    assert np.allclose(np.std(X_train[continuos_f]), np.std(act_X_train[continuos_f]))
    assert np.allclose(np.std(X_test[continuos_f]), np.std(act_X_test[continuos_f]))

    assert X_train.shape == act_X_train.shape
    assert X_test.shape == act_X_test.shape