import pandas as pd
import numpy as np
import os
import sys
import json
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg

@task
def load_data():
    df = pd.read_csv(cfg.DATA_FOLDER + 'heart.csv')
    return df

@task
def split_data(df, target):
    train_X, test_X, train_y, test_y = train_test_split(df.drop([target], axis=1), df[target], test_size=0.25, random_state=cfg.SEED_VALUE, stratify=df[target])
    return train_X, test_X, train_y, test_y

@task
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

@task
def clean_data(X, y):
    index = X[X["Cholesterol"] == 0].index
    X.drop(index, axis=0, inplace=True)
    y.drop(index, axis=0, inplace=True)
    return X, y

@task
def encode_data(X_train, X_test, categorical_f):
    for dataset in [X_train, X_test]:
        dataset["Sex"] = dataset["Sex"].map({'M':1, 'F':0})
        dataset["ExerciseAngina"] = dataset["ExerciseAngina"].map({'N':0, 'Y':1})

    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train[categorical_f])

    for dataset in [X_train, X_test]:
        dataset[ohe.get_feature_names_out()] = ohe.transform(dataset[categorical_f]).toarray().astype('int8')
        dataset.drop(categorical_f, axis=1, inplace=True)

    return X_train, X_test, ohe

@task
def detect_VIF(df):
    df_ = df.copy()
    df_['intercept'] = 1
    with np.errstate(divide='ignore'):
        while(True):
            df_vif = pd.DataFrame(columns=["Features", "VIF"])
            df_vif["Features"] = df_.columns
            df_vif["VIF"] = [variance_inflation_factor(df_.values, i) for i in range(len(df_.columns))]
            df_vif = df_vif[df_vif["Features"] != "intercept"].sort_values("VIF", ascending=False)
            if df_vif.iloc[0]["VIF"] > 5:
                df_.drop([df_vif.iloc[0]["Features"]], axis=1, inplace=True)
            else:
                next_ = False
                break
    df_.drop(["intercept"], axis=1, inplace=True)
    return df, df_vif
        
@task
def oversample_data(X_train, y_train):
    oversampler = SMOTE(random_state=42, k_neighbors=10)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    return X_train, y_train, oversampler

@task
def scale_data(X_train, X_test, continuos_f):
    ss = StandardScaler()
    X_train[continuos_f] = ss.fit_transform(X_train[continuos_f])
    X_test[continuos_f] = ss.transform(X_test[continuos_f])
    return X_train, X_test, ss

@task
def feature_selection(X_train, y_train, X_test):
    fs_clf = SelectKBest(score_func=f_classif, k=15)
    X_train = fs_clf.fit_transform(X_train, y_train)
    X_test = fs_clf.transform(X_test)
    return X_train, X_test, fs_clf

def get_scores(y_true, y_pred, y_pred_proba):
    return {
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "ROC AUC": roc_auc_score(
            y_true, y_pred_proba, average="weighted", multi_class="ovr"
        ),
    }

@task
def training(X_train, y_train, X_test, y_test, metric="recall", cv=8, model_alias="challenger"):
    with mlflow.start_run():
        param_grid = [{
            'n_neighbors':np.arange(5, 21),
            'weights':['uniform', 'distance'],
            'p':[1, 2],
        }]

        knn_clf = KNeighborsClassifier()
        grid_knn = GridSearchCV(knn_clf, param_grid, cv=cv, scoring=["f1", metric], refit="f1")
        grid_knn.fit(X_train, y_train)
    
        y_pred = grid_knn.predict(X_test)
        y_pred_proba = grid_knn.predict_proba(X_test)[:, 1]
        scores = get_scores(y_test, y_pred, y_pred_proba)

        mlflow.log_param("params", grid_knn.get_params(deep=True))

        for metric in [
            "Balanced Accuracy",
            "F1 Score",
            "Precision",
            "Recall",
            "ROC AUC",
        ]:
            mlflow.log_metric(metric, scores[metric])

        mlflow.sklearn.log_model(
            sk_model=grid_knn,
            artifact_path="KNeighborsClassifier",
            registered_model_name=cfg.MODEL_NAME,
            signature=mlflow.models.infer_signature(X_train, grid_knn.predict(X_train)),
            input_example=X_train[0:2],
        )
        
        ohe_path = os.path.join(cfg.MODEL_FOLDER, "ohe.pkl")
        fs_vif_path = os.path.join(cfg.MODEL_FOLDER, "fs_vif.json")
        ovs_path = os.path.join(cfg.MODEL_FOLDER, "ovs.pkl")
        ss_path = os.path.join(cfg.MODEL_FOLDER, "ss.pkl")
        fs_path = os.path.join(cfg.MODEL_FOLDER, "fs.pkl")

        mlflow.log_artifact(ohe_path, artifact_path="preprocessing")
        mlflow.log_artifact(fs_vif_path, artifact_path="preprocessing")
        mlflow.log_artifact(ovs_path, artifact_path="preprocessing")
        mlflow.log_artifact(ss_path, artifact_path="preprocessing")
        mlflow.log_artifact(fs_path, artifact_path="preprocessing")
        
        os.remove(ohe_path)
        os.remove(fs_vif_path)
        os.remove(ovs_path)
        os.remove(ss_path)
        os.remove(fs_path)

        client = MlflowClient()

        latest_mv = client.get_latest_versions(cfg.MODEL_NAME, stages=["None"])[0]
        client.set_registered_model_alias(
            cfg.MODEL_NAME, model_alias, latest_mv.version
        )

        client.set_model_version_tag(
            name=cfg.MODEL_NAME,
            version=latest_mv.version,
            key="task",
            value="classification",
        )

        for metric in [
            "Balanced Accuracy",
            "F1 Score",
            "Precision",
            "Recall",
            "ROC AUC",
        ]:
            client.set_model_version_tag(
                name=cfg.MODEL_NAME,
                version=latest_mv.version,
                key=metric.replace(" ", "_").lower(),
                value=str(scores[metric]),
            )

    return grid_knn

@task
def upload_model_artifacts_to_gcs(project_id, bucket_name, local_dir, prefix=""):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            blob_path = os.path.join(
                prefix, os.path.relpath(local_path, local_dir)
            ).replace("\\", "/")
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")
    return

@flow(name="Cardiovascular Diseases ML Pipeline", retries=1, retry_delay_seconds=300)
def cardiovascular_diseases_pipeline(model_alias):
    df = load_data()

    continuos_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_f = ["ChestPainType", "RestingECG", "ST_Slope"]
    target = "HeartDisease"

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(df, target)
    
    X_train, X_test, _ = detect_outliers(X_train, ["RestingBP", "Cholesterol", "Oldpeak"], "leve", X_test)
    X_train, y_train = clean_data(X_train, y_train)
    X_train, X_test, ohe = encode_data(X_train, X_test, categorical_f)
    X_train, _ = detect_VIF(X_train)
    selected_features = X_train.columns.tolist()
    X_train, y_train, ovs = oversample_data(X_train, y_train)
    X_train, X_test, ss = scale_data(X_train, X_test, continuos_f)
    X_train, X_test, fs = feature_selection(X_train, y_train, X_test)

    with open(cfg.MODEL_FOLDER + "ohe.pkl", "wb") as f:
        cloudpickle.dump(ohe, f)

    with open(cfg.MODEL_FOLDER + "fs_vif.json", "w") as f:
        json.dump(selected_features, f)

    with open(cfg.MODEL_FOLDER + "ovs.pkl", "wb") as f:
        cloudpickle.dump(ovs, f)

    with open(cfg.MODEL_FOLDER + "ss.pkl", "wb") as f:
        cloudpickle.dump(ss, f)

    with open(cfg.MODEL_FOLDER + "fs.pkl", "wb") as f:
        cloudpickle.dump(fs, f)

    training(X_train, y_train, X_test, y_test, model_alias=model_alias)
    upload_model_artifacts_to_gcs(
        project_id="plucky-haven-463121-j1",
        bucket_name="plucky-haven-463121-j1-mlflow-models",
        local_dir="models",
        prefix="run_artifacts",
    )

if __name__ == "__main__":

    cfg.init_conn_mlflow()

    alias = sys.argv[1]
    cardiovascular_diseases_pipeline(model_alias=alias)

# make run-training ALIAS=champion