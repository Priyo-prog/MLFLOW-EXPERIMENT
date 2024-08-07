import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# Current Directory
CURRENT_DIRECTORY = Path().absolute()
DATA_DIRECTORY = CURRENT_DIRECTORY/"Data"


# Load the dataset
def load_data():
    return pd.read_csv(Path(f"{DATA_DIRECTORY}/winequality-red.csv"), sep=";")

def eval_function(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main(alpha, l1_ratio):
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
                                    df.drop(columns=["quality"], axis=1),
                                    df["quality"], test_size=0.2, random_state=42)
    

    mlflow.set_experiment("ML-Model-1")
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_function(y_test, y_pred)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2-SCORE", r2)
        mlflow.sklearn.log_model(model, "trained_model") # model, folder name
        os.makedirs("model", exist_ok=True)
        joblib.dump(value=model, filename=Path("model/regression_model.pkl"))
        mlflow.log_artifact( "model", "Regression Model")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alpha", "-a", type=float, default=0.2) 
    args.add_argument("--l1_ratio", "-l1", type=float, default=0.3)       
    parsed_args = args.parse_args()

    # Now pass the arguments to the main function
    main(parsed_args.alpha, parsed_args.l1_ratio)



    
  

