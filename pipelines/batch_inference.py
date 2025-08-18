import pandas as pd
import sys
import os
from prefect import flow, task
from google.cloud import storage
import cloudpickle
from mlflow.tracking import MlflowClient
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg

cfg.init_conn_mlflow()


@task
def load_data(filename):
    test = pd.read_csv(filename)
    return test


def split_X_y(test, target):
    return test.drop(target, axis=1), test[target]


@task
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


@task
def prepare_data(test):
    continuos_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_f = ["ChestPainType", "RestingECG", "ST_Slope"]
    target = "HeartDisease"

    X_test, y_test = split_X_y(test, target)
    X_test = transform_data(X_test, continuos_f, categorical_f)

    return X_test, y_test


@task
def apply_model(X_test):
    y_pred = model.predict(X_test)
    return y_pred


@task
def make_result(df, y_pred):
    df["prediction"] = y_pred
    df_result = df[["prediction"]].copy()
    return df_result


@task
def save_result(df_result, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = f"{output_folder}/predictions.parquet"
    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    return


def upload_blob(project_id, bucket_name, source_file_name, destination_blob_name):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded as {destination_blob_name}")
    return


@task
def upload2cloud(project_id, bucket_name, output_folder):
    filename = f"{output_folder}/predictions.parquet"
    upload_blob(
        project_id=project_id,
        bucket_name=bucket_name,
        source_file_name=filename,
        destination_blob_name=filename,
    )
    return


@flow(name="Obesity Level Inference Pipeline", retries=1, retry_delay_seconds=300)
def cardiovascular_diseases_inference_pipeline(project_id, bucket_name, filepath):
    output_folder = f"output"

    df = load_data(filepath)
    X_test, _ = prepare_data(df)
    y_pred = apply_model(X_test)
    df_result = make_result(df, y_pred)
    save_result(df_result, output_folder)
    upload2cloud(project_id, bucket_name, output_folder)
    return


if __name__ == "__main__":
    alias = sys.argv[1]
    filepath = sys.argv[2]

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(cfg.MODEL_NAME, alias)
    RUN_ID = model_version.run_id
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

    project_id = "plucky-haven-463121-j1"
    bucket_name = "plucky-haven-463121-j1-predictions"
    cardiovascular_diseases_inference_pipeline(project_id, bucket_name, filepath)

# make run-inference ALIAS=champion
