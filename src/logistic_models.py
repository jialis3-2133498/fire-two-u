import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


def create_horizon_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["y_12"] = (
        (df["event"] == 1) & (df["time_to_hit_hours"] <= 12)).astype(int)
    df["y_24"] = (
        (df["event"] == 1) & (df["time_to_hit_hours"] <= 24)).astype(int)
    df["y_48"] = (
        (df["event"] == 1) & (df["time_to_hit_hours"] <= 48)).astype(int)
    df["y_72"] = (
        (df["event"] == 1) & (df["time_to_hit_hours"] <= 72)).astype(int)

    return df


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
    return model, imputer, scaler, auc, y_valid, y_prob


def train_all_logistic_models(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str]):
    model_dict = {}
    for each_target_col in target_cols:
        model, imputer, scaler, auc, y_valid, y_prob = train_logistic_model(
            train_df, valid_df, feature_cols, each_target_col)
        model_dict[each_target_col] = {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "auc": auc,
            "y_valid": y_valid,
            "y_prob": y_prob
        }
    return model_dict


def predict_logistic_model(
        model,
        imputer,
        scaler,
        df: pd.DataFrame,
        feature_cols: list[str]
):
    X = df[feature_cols]
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    return y_prob


def predict_all_horizons(
        fitted_models: dict,
        df: pd.DataFrame,
        feature_cols: list[str]
):
    prediction_dict = {}
    for each_key in fitted_models.keys():
        each_model = fitted_models[each_key]["model"]
        each_imputer = fitted_models[each_key]["imputer"]
        each_scaler = fitted_models[each_key]["scaler"]
        y_prob = predict_logistic_model(each_model,
                                        each_imputer,
                                        each_scaler,
                                        df,
                                        feature_cols)
        prediction_dict[each_key] = y_prob
    return prediction_dict


def evaluate_predictions(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, output_dict=True)
    return auc, report
