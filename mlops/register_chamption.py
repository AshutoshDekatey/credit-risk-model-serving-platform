import pickle

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "credit-risk-model"


# Load the exact artifact currently deployed in Kubernetes
with open("model/credit_risk_model.pkl", "rb") as file:
    model = pickle.load(file)


# Fixed synthetic validation set.
# We will use this same dataset when evaluating the challenger.
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
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
])


mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("credit-risk-model-lifecycle")


predictions = model.predict(X_val)

metrics = {
    "accuracy": accuracy_score(y_val, predictions),
    "precision": precision_score(y_val, predictions, zero_division=0),
    "recall": recall_score(y_val, predictions, zero_division=0),
    "f1": f1_score(y_val, predictions, zero_division=0),
}


signature = infer_signature(X_val, predictions)


with mlflow.start_run(run_name="production-baseline-v1") as run:

    mlflow.log_params({
        "model_family": "LogisticRegression",
        "source_release": "v0.1.0",
        "feature_count": 4,
    })

    mlflow.log_metrics(metrics)

    mlflow.set_tags({
        "role": "production-baseline",
        "environment": "development",
        "source": "session-1-production-artifact",
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
    alias="champion",
    version=model_version.version,
)


print(f"Run ID: {run_id}")
print(f"Registered model: {REGISTERED_MODEL_NAME}")
print(f"Model version: {model_version.version}")
print("Alias: champion")

print("\nValidation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")
