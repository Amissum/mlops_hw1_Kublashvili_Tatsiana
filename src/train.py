
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

def main():
    params = yaml.safe_load(open("params.yaml"))
    train_params = params.get("train", {})
    alpha = train_params.get("alpha", 0.1)
    test_size = train_params.get("test_size", 0.2)
    random_state = train_params.get("random_state", 42)

    df = pd.read_csv("data/train.csv")
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("dvc_pipeline_experiment")

    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

    with open("metrics.json", "w") as f:
        f.write('{"mse": %f}' % mse)

if __name__ == "__main__":
    main()
