from mlflow.tracking import MlflowClient
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg

cfg.init_conn_mlflow()

client = MlflowClient()

champion_version = client.get_model_version_by_alias(cfg.MODEL_NAME, "champion").version
challenger_version = client.get_model_version_by_alias(
    cfg.MODEL_NAME, "challenger"
).version

champion_tags = client.get_model_version(cfg.MODEL_NAME, champion_version).tags
challenger_tags = client.get_model_version(cfg.MODEL_NAME, challenger_version).tags

champion_roc = float(champion_tags.get("roc_auc", 0))
challenger_roc = float(challenger_tags.get("roc_auc", 0))

if challenger_roc > champion_roc:
    print(f"Promoting challenger (v{challenger_version}) to champion")
    client.delete_registered_model_alias(cfg.MODEL_NAME, "champion")
    client.delete_registered_model_alias(cfg.MODEL_NAME, "challenger")
    client.set_registered_model_alias(cfg.MODEL_NAME, "champion", challenger_version)
    client.set_model_version_tag(cfg.MODEL_NAME, champion_version, "archived", "true")
    print(
        f"Version {challenger_version} is now 'champion'. Version {champion_version} archived."
    )
else:
    print("Challenger did not outperform champion. No promotion executed.")
