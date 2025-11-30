
import os
import yaml
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    params = yaml.safe_load(open("params.yaml"))
    prep_params = params.get("prepare", {})
    test_size = prep_params.get("test_size", 0.2)
    random_state = prep_params.get("random_state", 42)
    stratify_flag = prep_params.get("stratify", True)

    iris = datasets.load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "label"})

    X = df.drop(columns=["label"])
    y = df["label"]

    stratify = y if stratify_flag else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_df = X_train.copy()
    train_df["label"] = y_train

    test_df = X_test.copy()
    test_df["label"] = y_test

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Saved data/train.csv and data/test.csv")

if __name__ == "__main__":
    main()
