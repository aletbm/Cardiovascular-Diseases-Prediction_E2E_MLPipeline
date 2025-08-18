import os
import sys
from dotenv import load_dotenv
import mlflow
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "cardiovascular_experiment"
MODEL_NAME = "MyKNNClassifier"
DRIFT_THRESHOLD = 0.3
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SEED_VALUE = 42

INPUT_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(INPUT_FOLDER, "data/")
MODEL_FOLDER = os.path.join(INPUT_FOLDER, "models/")
REPORT_FOLDER = "./monitoring/"

def init_conn_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return

os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
