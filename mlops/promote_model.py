import mlflow
from mlflow import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "credit-risk-model"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

challenger = client.get_model_version_by_alias(
    MODEL_NAME,
    "challenger"
)

print(f"Current champion: Version {champion.version}")
print(f"Candidate challenger: Version {challenger.version}")


# Preserve where production came from before moving the champion pointer.
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="previous_champion",
    version=champion.version,
)

# Approve challenger.
client.set_model_version_tag(
    name=MODEL_NAME,
    version=challenger.version,
    key="validation_status",
    value="approved",
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=challenger.version,
    key="deployment_status",
    value="approved_for_deployment",
)

# Promote V2.
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=challenger.version,
)

# Challenger has completed its lifecycle.
client.delete_registered_model_alias(
    name=MODEL_NAME,
    alias="challenger",
)

# Mark old champion.
client.set_model_version_tag(
    name=MODEL_NAME,
    version=champion.version,
    key="deployment_status",
    value="superseded",
)


new_champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

previous_champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "previous_champion"
)

print()
print("Promotion complete.")
print(f"champion          -> Version {new_champion.version}")
print(f"previous_champion -> Version {previous_champion.version}")
