import pandas as pd
import numpy as np
import os
import pickle
import json
from dvclive import Live
import yaml

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

n_estimators = yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]
# test_processed = pd.read_csv("./data/processed/test_processed.csv")
def load_data(filepath:str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)

    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")


# X_test = test_processed.iloc[:, :-1].values
# y_test = test_processed.iloc[:, -1].values
# X_test = test_processed.drop(columns=['Potability'])
# y_test = test_processed['Potability']
def prepare_data(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = df.drop(columns=["Potability"])
        y = df["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data : {e}")


# model = pickle.load(open("model.pkl", "rb"))
def load_model(model_path:str):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {model_path} :  {e}")


# y_pred = model.predict(X_test)
def evualation_model(model, X_test : pd.DataFrame, y_test : pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        metrics_dict = {
            'acc': acc,
            'pre': pre,
            'recall': recall,
            'f1_score': f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error Evaluating model : {e}")


# with open('metrics.json', 'w') as file:
#     json.dump(metrics_dict, file, indent=4)
def save_metrics(metrics_dict:dict, filepath:str)->None:
    try:
        with open(filepath, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath} : {e}")


def main():
    try:
        test_data_path = "./data/processed/test_processed_median.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evualation_model(model, X_test, y_test)
        with Live(save_dvc_exp=True) as live:
            live.log_metric("acc", metrics["acc"])
            live.log_metric("pre", metrics["pre"])
            live.log_metric("recall", metrics["recall"])
            live.log_metric("f1_score", metrics["f1_score"])
            live.log_param("n_estimators", n_estimators)

        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f"Error occurred : {e}")


if __name__ == "__main__":
    main()