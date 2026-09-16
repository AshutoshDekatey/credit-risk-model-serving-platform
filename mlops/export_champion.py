import json
import pickle

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "credit-risk-model"
ALIAS ="champion"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    ALIAS,
)

model_uri =f"models:/{MODEL_NAME}@{ALIAS}"

model = mlflow.sklearn.load_model(model_uri)

with open("model/credit_risk_model.pkl","wb") as file:
    pickle.dump(model,file)

metadata = {
    "registered_model":MODEL_NAME,
    "model_version":model_version.version,
    "alias":ALIAS,
    "run_id":model_version.run_id,
    "release":"v0.2.0",
}

with open("model/model_metadata.json","w") as file:
    json.dump(metadata,file, indent=2)

print("Champion exported.")
print(f"Registered model: {MODEL_NAME}")
print(f"Version: {model_version.version}")
print(f"Alias: {ALIAS}")
print("Serving artifact: model/credit_risk_model.pkl")
