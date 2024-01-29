import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from prefect import flow, task


@task(retries=3, retry_delay_seconds=2)
def read_dataframe(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    lab = pd.read_parquet(filename)

    lab.lpep_pickup_datetime = pd.to_datetime(lab.lpep_pickup_datetime)    #datetime - to tell pandas that it is not a string but date
    lab.lpep_dropoff_datetime = pd.to_datetime(lab.lpep_dropoff_datetime)

    lab['duration'] = lab.lpep_dropoff_datetime - lab.lpep_pickup_datetime   # adding duration to the list
    lab.duration = lab.duration.apply(lambda td: td.total_seconds() / 60)

    lab = lab[((lab.duration >= 1) & (lab.duration <= 60))]

    categ = ['PULocationID', 'DOLocationID']
    lab[categ] = lab[categ].astype(str)

    return lab


@task
def add_features(
    lab_train: pd.DataFrame, lab_val: pd.DataFrame
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Add features to the model"""
    lab_train["PU_DO"] = lab_train["PULocationID"] + "_" + lab_train["DOLocationID"]
    lab_val["PU_DO"] = lab_val["PULocationID"] + "_" + lab_val["DOLocationID"]

    categ = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    num = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categ + num].to_dict(orient="records")
    x_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categ + num].to_dict(orient="records")
    x_val = dv.transform(val_dicts)

    y_train = lab_train["duration"].values
    y_val = lab_val["duration"].values
    return x_train, x_val, y_train, y_val, dv


@task(log_prints=True)
def train_best_model(
    x_train: scipy.sparse._csr.csr_matrix,
    x_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(x_train, label=y_train)
        valid = xgb.DMatrix(x_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
    return None


@flow
def main_flow(
    train_path: str = "green_tripdata_2021-01.parquet",
    val_path: str = "green_tripdata_2021-02.parquet",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('Learn MLFLOW')

    # Load
    lab_train = read_dataframe(train_path)
    lab_val = read_dataframe(val_path)

    # Transform
    x_train, x_val, y_train, y_val, dv = add_features(lab_train, lab_val)

    # Train
    train_best_model(x_train, x_val, y_train, y_val, dv)


if __name__ == "__main__":
    main_flow()