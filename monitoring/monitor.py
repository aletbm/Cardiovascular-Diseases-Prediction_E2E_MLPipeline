import pandas as pd
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from evidently import Report, Dataset, DataDefinition
from evidently.presets import (
    DataDriftPreset,
    ClassificationPreset,
    DataSummaryPreset,
)
from evidently import MulticlassClassification
from sklearn.model_selection import train_test_split
import prefect
from prefect import flow, task
import os
import requests
import sys
import json
from datetime import datetime
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines.training_flow import (
    cardiovascular_diseases_pipeline,
    load_data,
)
import config as cfg

cfg.init_conn_mlflow()

ARTIFACT_DIR = "monitoring/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


@task
def load_model_and_artifacts(model_alias):
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(cfg.MODEL_NAME, model_alias)
    run_id = model_version.run_id

    model = mlflow.pyfunc.load_model(f"models:/{cfg.MODEL_NAME}@{model_alias}")
    ohe_path = client.download_artifacts(
        run_id, "preprocessing/ohe.pkl", ARTIFACT_DIR
    )
    fs_vif_path = client.download_artifacts(
        run_id, "preprocessing/fs_vif.json", ARTIFACT_DIR
    )
    ss_path = client.download_artifacts(
        run_id, "preprocessing/ss.pkl", ARTIFACT_DIR
    )
    fs_path = client.download_artifacts(
        run_id, "preprocessing/fs.pkl", ARTIFACT_DIR
    )

    with open(ohe_path, "rb") as f:
        ohe = cloudpickle.load(f)

    with open(fs_vif_path, 'r') as f:
        selected_features = json.load(f)

    with open(ss_path, "rb") as f:
        ss = cloudpickle.load(f)

    with open(fs_path, "rb") as f:
        fs = cloudpickle.load(f)

    return model, ohe, selected_features, ss, fs


@task
def split_data(df, target):
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=cfg.SEED_VALUE, stratify=df[target])
    return df_train, df_test


@task
def prepare_datasets(model, ohe, selected_features, ss, fs, train_df, test_df, target_col="HeartDisease"):
    continuos_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_f = ["ChestPainType", "RestingECG", "ST_Slope"]

    def make_dataset(df):
        X = df.drop(columns=[target_col], axis=1).copy()
        y = df[target_col].copy()

        X["Sex"] = X["Sex"].map({'M':1, 'F':0})
        X["ExerciseAngina"] = X["ExerciseAngina"].map({'N':0, 'Y':1})

        X[ohe.get_feature_names_out()] = ohe.transform(X[categorical_f]).toarray().astype('int8')
        X.drop(categorical_f, axis=1, inplace=True)
        
        X = X[selected_features].copy()
        X[continuos_f] = ss.transform(X[continuos_f])
        X = fs.transform(X)
        preds = model.predict(X)

        df2 = pd.DataFrame(X, columns=fs.get_feature_names_out())
        df2["target"] = y.tolist()
        df2["prediction"] = preds

        numerical_features = df2.select_dtypes(include=[int, float]).columns.tolist()

        return Dataset.from_pandas(
            df2,
            data_definition=DataDefinition(
                classification=[
                    MulticlassClassification(
                        target="target", prediction_labels="prediction"
                    )
                ],
                numerical_columns=numerical_features,
            ),
        )

    ds_train = make_dataset(train_df)
    ds_test = make_dataset(test_df)
    return ds_train, ds_test


@task
def run_monitoring(ds_train, ds_test):
    report = Report(
        metrics=[DataDriftPreset(), DataSummaryPreset(), ClassificationPreset()]
    )
    result = report.run(reference_data=ds_train, current_data=ds_test)

    if not os.path.exists(cfg.REPORT_FOLDER):
        os.makedirs(cfg.REPORT_FOLDER)

    result.save_html(cfg.REPORT_FOLDER + "full_monitor_report.html")

    report_dict = result.json()
    report_dict = json.loads(report_dict)

    for metric in report_dict["metrics"]:
        metric_id = metric["metric_id"]
        value = metric["value"]

        if "DriftedColumnsCount" in metric_id:
            drift_score = value["share"]
            break

    print(f"Drift score: {drift_score}")
    return drift_score


@task
def send_slack_alert(message: str, state, flow_name):
    color = {
        "SUCCESS": "#36a64f",
        "FAILURE": "#ff0000",
        "INFO": "#3AA3E3",
        "WARNING": "#FFA500",
    }.get(state, "#cccccc")

    emoji = {"SUCCESS": "✅", "FAILURE": "❌", "INFO": "ℹ️", "WARNING": "⚠️"}.get(
        state, "ℹ️"
    )

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    template = {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{emoji} Prefect Flow Alert - {state}",
                            "emoji": True,
                        },
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Flow:* `{flow_name}`"},
                            {"type": "mrkdwn", "text": f"*Status:* *{state}*"},
                            {"type": "mrkdwn", "text": f"*Time:* {date}"},
                        ],
                    },
                ],
            }
        ]
    }

    if message:
        template["attachments"][0]["blocks"].append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Message:* `{message}`"},
            }
        )

    if not cfg.SLACK_WEBHOOK_URL:
        print("Slack webhook not configured.")
        return

    response = requests.post(cfg.SLACK_WEBHOOK_URL, json=template)
    if response.status_code == 200:
        print("Slack alert sent successfully.")
    else:
        print(f"Failed to send Slack alert: {response.status_code} {response.text}")


@task
def check_drift_and_maybe_retrain(drift_score: float, model_alias: str, flow_name: str):
    if drift_score > cfg.DRIFT_THRESHOLD:
        alert_msg = f"Drift detected (score={drift_score:.3f}) > threshold ({cfg.DRIFT_THRESHOLD})"
        print(alert_msg)
        send_slack_alert(alert_msg, state="WARNING", flow_name=flow_name)

        msg = "Starting model retraining..."
        print(msg)
        send_slack_alert(msg, state="INFO", flow_name=flow_name)

        cardiovascular_diseases_pipeline(model_alias=model_alias + "_retrain")

        msg = "Retraining completed."
        print(msg)
        send_slack_alert(msg, state="SUCCESS", flow_name=flow_name)
        return True
    else:
        msg = f"Drift is acceptable ({drift_score:.3f} ≤ {cfg.DRIFT_THRESHOLD})"
        print(msg)
        send_slack_alert(msg, state="SUCCESS", flow_name=flow_name)
        return False


@flow(name="Monitoring + Conditional Retraining")
def monitoring_flow(model_alias):
    flow_name = prefect.runtime.flow_run.name
    msg = "Monitoring started."
    send_slack_alert(msg, state="INFO", flow_name=flow_name)
    model, ohe, selected_features, ss, fs = load_model_and_artifacts(model_alias=model_alias)
    df = load_data()
    train_df, test_df = split_data(df, target="HeartDisease")
    ds_train, ds_test = prepare_datasets(model, ohe, selected_features, ss, fs, train_df, test_df, target_col="HeartDisease")
    drift_score = run_monitoring(ds_train, ds_test)
    check_drift_and_maybe_retrain(
        drift_score, model_alias=model_alias, flow_name=flow_name
    )


if __name__ == "__main__":
    alias = sys.argv[1]
    monitoring_flow(model_alias=alias)

# prefect server start
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models
# make run-monitoring ALIAS=champion