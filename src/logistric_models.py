import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def create_horizon_label(
        df: pd.DataFrame
) -> pd.DataFrame:
    return


def train_all_logistic_models(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str]
):
    return


def train_logistic_model(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        random_state: int = 42
        ):
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    # SimpleImputer() fills out missing value in the dataset
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)

    # StandardScaler() standardizes each feature so that they have
    # a similar scale, usually mean at 0, and std at 1.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_valid_scaled = scaler.transform(X_valid_imputed)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state)
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_valid_scaled)[:, 1]
    auc = roc_auc_score(y_valid, y_prob)
    return model, imputer, scaler, auc


def predict_logistic_model(
        model,
        imputer,
        scaler,
        df: pd.DataFrame,
        feature_cols: list[str]
):
    return


def predict_all_horizons(
        fitted_models: dict,
        df: pd.DataFrame,
        feature_cols: list[str]
):
    return


def evaluate_predictions(y_true, y_prob):
    return
