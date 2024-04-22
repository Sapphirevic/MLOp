import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import mlflow
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from prefect import flow, task

df = pd.read_csv("C:/Users/Victoria/gt/MLOp/Housing_Prediction/Training.csv")
df_train = pd.read_csv("C:/Users/Victoria/gt/MLOp/Housing_Prediction/Training.csv")
df_val = pd.read_csv("C:/Users/Victoria/gt/MLOp/Housing_Prediction/Validation.csv")


def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df.drop("Id", axis=1)
    df = df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"])

    df = df[df.Neighborhood == "Avondale"]
    df = df[(df.SalePrice >= 100000) & (df.SalePrice <= 500000)]

    categorical = ["BedroomAbvGr", "GarageCars"]
    df[categorical].astype(str)

    return df


def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Add features to the model"""
    df_train["BedroomAbvGr"] = df_train["BedroomAbvGr"].astype(str)
    df_train["GarageCars"] = df_train["GarageCars"].astype(str)

    df_train["BED_GAR"] = df_train["BedroomAbvGr"] + "_" + df_train["GarageCars"]

    df_val["BedroomAbvGr"] = df_val["BedroomAbvGr"].astype(str)
    df_val["GarageCars"] = df_val["GarageCars"].astype(str)

    df_val["BED_GAR"] = df_val["BedroomAbvGr"] + "_" + df_val["GarageCars"]

    categorical = ["BED_GAR"]  #'BedroomAbvGr', 'GarageCars']
    numerical = ["GarageArea"]

    dv = DictVectorizer()

    train_dict = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dict)

    y_train = df_train["SalePrice"].values
    y_val = df_val["SalePrice"].values
    return X_train, X_val, y_train, y_val, dv


def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """train a model with best hyperparams and write everything out"""
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.06817254433679483,
            "max_depth": 4,
            "min_child_weight": 5.342622580585712,
            "objective": "reg:linear",
            "reg_alpha": 0.03767824771170366,
            "reg_lambda": 0.11142683044900038,
            "seed": 42,
        }
        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
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


def main_flow(
    train_path: str = "C:/Users/Victoria/gt/MLOp/Housing_Prediction/Training.csv",
    val_path: str = "C:/Users/Victoria/gt/MLOp/Housing_Prediction/Validation.csv",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("First Prediction")

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv)


if __name__ == "__main__":
    main_flow()
