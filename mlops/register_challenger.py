import mlflow
import mlflow.sklearn
import numpy as np

from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "credit-risk-model"


# Expanded synthetic training dataset from the Data Science team.
# Features:
# income, debt_ratio, credit_score, missed_payments
X_train = np.array([
    [90000, 0.15, 780, 0],
    [75000, 0.20, 740, 0],
    [60000, 0.30, 700, 1],
    [45000, 0.40, 650, 2],
    [35000, 0.55, 590, 4],
    [30000, 0.65, 540, 6],
    [120000, 0.10, 820, 0],
    [50000, 0.35, 680, 1],
    [40000, 0.50, 610, 3],
    [25000, 0.70, 520, 7],

    [100000, 0.18, 790, 0],
    [82000, 0.25, 730, 0],
    [68000, 0.28, 710, 1],
    [52000, 0.32, 690, 1],
    [47000, 0.22, 630, 0],
    [32000, 0.58, 600, 4],
    [29000, 0.62, 560, 5],
    [76000, 0.55, 590, 4],
    [38000, 0.60, 605, 5],
    [55000, 0.57, 615, 4],
    [42000, 0.18, 640, 0],
    [31000, 0.33, 705, 2],
    [88000, 0.48, 610, 3],
    [65000, 0.52, 625, 3],
    [27000, 0.40, 660, 3],
    [95000, 0.42, 670, 2],
])

y_train = np.array([
    0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 0, 0
])


# IMPORTANT:
# Same validation set used for Version 1.
X_val = np.array([
    [85000, 0.22, 760, 0],
    [28000, 0.68, 530, 6],
    [35000, 0.20, 750, 0],
    [70000, 0.65, 580, 5],
    [55000, 0.60, 620, 4],
    [45000, 0.20, 620, 0],
    [80000, 0.50, 600, 3],
    [30000, 0.30, 700, 2],
    [42000, 0.58, 610, 4],
    [50000, 0.25, 710, 0],
])

y_val = np.array([
    0, 1, 0, 1, 1,
    0, 1, 0, 1, 0
])


model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    ),
])


model.fit(X_train, y_train)

predictions = model.predict(X_val)

metrics = {
    "accuracy": accuracy_score(y_val, predictions),
    "precision": precision_score(y_val, predictions, zero_division=0),
    "recall": recall_score(y_val, predictions, zero_division=0),
    "f1": f1_score(y_val, predictions, zero_division=0),
}


mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("credit-risk-model-lifecycle")

signature = infer_signature(X_val, predictions)


with mlflow.start_run(run_name="challenger-v2") as run:

    mlflow.log_params({
        "model_family": "LogisticRegression",
        "feature_scaling": "StandardScaler",
        "class_weight": "balanced",
        "training_samples": len(X_train),
        "feature_count": 4,
    })

    mlflow.log_metrics(metrics)

    mlflow.set_tags({
        "role": "challenger",
        "source": "data-science-candidate",
        "candidate_release": "v0.2.0",
    })

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_val[:2],
    )

    run_id = run.info.run_id


model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=REGISTERED_MODEL_NAME,
)


client = MlflowClient()

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="challenger",
    version=model_version.version,
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=model_version.version,
    key="validation_status",
    value="awaiting_approval",
)


print(f"Run ID: {run_id}")
print(f"Registered model: {REGISTERED_MODEL_NAME}")
print(f"Model version: {model_version.version}")
print("Alias: challenger")

print("\nValidation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")
