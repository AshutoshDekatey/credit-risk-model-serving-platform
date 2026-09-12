# Credit Risk Model Serving Platform

A containerized ML inference service deployed on Kubernetes with replicated model-serving instances and self-healing orchestration.

## Project Objective

A trained credit-risk model originally exists as a Python model artifact. The objective is to convert that artifact into a network-accessible inference service and deploy it as a replicated workload on Kubernetes.

The system demonstrates the transition from:

**trained model → inference API → Docker image → Kubernetes Deployment → replicated serving service**

## Architecture

```text
Loan Application
       |
       | HTTP / JSON
       v
Kubernetes Service
       |
       +----------------+----------------+
       |                |                |
       v                v                v
 Model API Pod     Model API Pod     Model API Pod
       |                |                |
       +----------------+----------------+
                        |
                        v
             Credit Risk ML Model
```

## Technology

* Python
* scikit-learn
* FastAPI
* Docker
* Kubernetes
* kind
* kubectl
* WSL 2

## Model API

### Health

`GET /health`

Example response:

```json
{
  "status": "healthy"
}
```

### Prediction

`POST /predict`

Example request:

```json
{
  "income": 85000,
  "debt_ratio": 0.22,
  "credit_score": 760,
  "missed_payments": 0
}
```

Example response:

```json
{
  "prediction": 0,
  "risk": "LOW"
}
```

Incoming requests are validated before inference. Invalid feature types are rejected by the API rather than passed directly to the model.

## Containerization

The application, Python runtime, dependencies and trained model are packaged into the Docker image:

```text
credit-risk-api:v0.1.0
```

The container exposes the model-serving API on port `8000`.

## Kubernetes Deployment

The application is deployed using a Kubernetes Deployment configured with three replicas.

```text
Desired replicas: 3
```

A Kubernetes Service provides a stable endpoint in front of the model-serving Pods.

For local development, the Service can be accessed using:

```bash
kubectl port-forward service/credit-risk-api 8080:8000
```

## Self-Healing Demonstration

A running model-serving Pod was manually deleted:

```bash
kubectl delete pod <pod-name>
```

Kubernetes detected that the actual replica count no longer matched the desired state and automatically created a replacement Pod.

The experiment was repeated with the replacement Pod.

After both failures:

```text
READY        3/3
UP-TO-DATE   3
AVAILABLE    3
```

Model inference remained available during the observed test.

This demonstrates Kubernetes' reconciliation-based workload management and automatic restoration of the desired replica count.

## Run Locally

Build the image:

```bash
docker build -t credit-risk-api:v0.1.0 .
```

Create the Kubernetes cluster:

```bash
kind create cluster --name credit-risk
```

Load the locally built image:

```bash
kind load docker-image credit-risk-api:v0.1.0 --name credit-risk
```

Deploy:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Verify:

```bash
kubectl get pods
kubectl get deployment
kubectl get service
```

Access locally:

```bash
kubectl port-forward service/credit-risk-api 8080:8000
```

## Current Scope

`v0.1.0` demonstrates local Kubernetes-based model serving and orchestration.

It does not yet represent a production-grade ML platform. Future versions can add health probes, observability, resource management, autoscaling, CI/CD, model registry integration, cloud deployment and controlled model releases.
