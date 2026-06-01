import argparse
import os
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def simulate_data(
    n_samples: int = 1000, n_features: int = 10, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, random_state=random_state)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return df, pd.Series(y, name="target")


def train_model(
    df: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
    model_name: str = "baseline_model",
    register: bool = False,
) -> dict:
    X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=random_state)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    # local save
    os.makedirs("models", exist_ok=True)
    local_path = os.path.join("models", "model_final.pkl")
    joblib.dump(model, local_path)

    # mlflow log
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "default"))
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("val_accuracy", float(acc))
        # log model as artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        # register model if requested and tracking server supports registry
        if register and model_name:
            try:
                mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)
                registered = True
            except Exception:
                registered = False
        else:
            registered = False

    return {"accuracy": float(acc), "local_path": local_path, "registered": registered}


def train_main():
    parser = argparse.ArgumentParser(description="Train baseline ML model and log to MLflow")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--model-name", type=str, default="baseline_model")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df, y = simulate_data(n_samples=args.n_samples, n_features=args.n_features, random_state=args.random_state)
    result = train_model(
        df,
        y,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        model_name=args.model_name,
        register=args.register,
    )
    print(
        f"Training complete. val_accuracy={result['accuracy']}, local_path={result['local_path']}, registered={result['registered']}"
    )


def predict_main():
    parser = argparse.ArgumentParser(description="Simple predict CLI using registered model or local model")
    parser.add_argument("--model-name", type=str, default="baseline_model")
    parser.add_argument("--model-version", type=str, default="Production")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=10)
    args = parser.parse_args()

    # try load from registry
    try:
        model_uri = f"models:/{args.model_name}/{args.model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception:
        # fallback to local
        local_path = os.path.join("models", "model_final.pkl")
        if not os.path.exists(local_path):
            raise FileNotFoundError("No local model found and failed to load from registry")
        model = joblib.load(local_path)

    X_fake = np.random.RandomState(0).rand(args.n_samples, args.n_features)
    df = pd.DataFrame(X_fake, columns=[f"f{i}" for i in range(args.n_features)])
    preds = model.predict(df)
    print(pd.DataFrame({"prediction": preds}))


if __name__ == "__main__":
    train_main()
