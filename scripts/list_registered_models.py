from mlflow.tracking import MlflowClient
from pprint import pprint
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg


# --- Configuration ---
client = MlflowClient(tracking_uri=cfg.MLFLOW_TRACKING_URI)


def list_all_models():
    for rm in client.search_registered_models():
        for alias in rm.aliases:
            pprint(alias)


if __name__ == "__main__":
    list_all_models()
