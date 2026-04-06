import pandas as pd
import numpy as np
import os

import pickle

from sklearn.ensemble import RandomForestClassifier
import yaml


# n_estimators = yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]
def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error Loading params from {params_path} :  {e}")


# train_processed = pd.read_csv("./data/processed/train_processed.csv")
def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from file {filepath} : {e}")


# X_train = train_processed.iloc[:, 0:-1].values
# y_train = train_processed.iloc[:, -1].values
# X_train = train_processed.drop(columns=['Potability'])
# y_train = train_processed['Potability']
def prepare_data(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = df.drop(columns=["Potability"])
        y = df["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data : {e}")



# clf = RandomForestClassifier(n_estimators=n_estimators)
# clf.fit(X_train, y_train)
def train_model(X:pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f"Error Training model : {e}")


# pickle.dump(clf, open("model.pkl", "wb"))
def save_model(model : RandomForestClassifier, filepath: str) ->None:
    try:
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {filepath}: {e}")


def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_processed_median.csv"
        model_name = "models/model.pkl"

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)
        model = train_model(X_train, y_train, n_estimators)

        save_model(model, model_name)
    except Exception as e:
        raise Exception(f"Error occured : {e}")

if __name__ == "__main__":
    main()