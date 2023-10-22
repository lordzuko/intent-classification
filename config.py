from __future__ import annotations

import logging
import os

import joblib
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
print(os.path.abspath(ENV_PATH))
load_dotenv(dotenv_path=ENV_PATH)

API_VERSION = "v1.0.0"
RANDOM_SEED = 42
MAX_TOKEN_COUNT = 100
ML_BERT_MODEL_NAME = "bert-base-multilingual-cased"


EXPECTED_LABELS = [
    "abbreviation",
    "aircraft",
    "airfare",
    "airline",
    "airport",
    "capacity",
    "cheapest",
    "city",
    "distance",
    "flight",
    "flight_no",
    "flight_time",
    "ground_fare",
    "ground_service",
    "meal",
    "quantity",
    "restriction",
]
# LABELWISE_THRESHOLDS = [
#     0.60,
#     0.60,
#     0.2,
#     0.30,
#     0.30,
#     0.2,
#     0.2,
#     0.30,
#     0.1,
#     0.2,
#     0.1,
#     0.2,
#     0.1,
#     0.30,
#     0.2,
#     0.30,
#     0.2,
# ]

PORT = int(os.environ["PORT"])
CHECKPOINT_PATH = os.environ["CHECKPOINT_PATH"]
LABEL_COLUMNS = joblib.load(os.environ["ML_BINARIZER_PATH"]).classes_
assert (
    list(LABEL_COLUMNS) == EXPECTED_LABELS
), "The order of labels are changed, please ensure they match."
# assert len(LABELWISE_THRESHOLDS) == len(
#     EXPECTED_LABELS
# ), "The count of LABELWISE_TRHESHOLDS should be equal to EXPECTED_LABELS"
