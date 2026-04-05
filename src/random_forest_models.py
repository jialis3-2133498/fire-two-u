import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def train_random_forest_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int = 42,
):
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    # Random Forest doesn't need scaling, but still needs imputation
    # Random Forest splits on thresholds, so feature scale doesn't matter.
    # Removed it to keep the pipeline simpler.
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,  # let trees grow fully
        min_samples_leaf=5,  # avoid overfitting on small leaves
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train_imputed, y_train)

    y_prob = model.predict_proba(X_valid_imputed)[:, 1]
    auc = roc_auc_score(y_valid, y_prob)
    return model, imputer, auc, y_valid, y_prob


def train_all_random_forest_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
):
    model_dict = {}
    for each_target_col in target_cols:
        model, imputer, auc, y_valid, y_prob = train_random_forest_model(
            train_df, valid_df, feature_cols, each_target_col
        )
        model_dict[each_target_col] = {
            "model": model,
            "imputer": imputer,
            "auc": auc,
            "y_valid": y_valid,
            "y_prob": y_prob
            }
        print(f"  [{each_target_col}] AUC: {auc:.4f}")
    return model_dict


def predict_random_forest_model(
    model, imputer, df: pd.DataFrame, feature_cols: list[str]
):
    X = df[feature_cols]
    X_imputed = imputer.transform(X)
    y_prob = model.predict_proba(X_imputed)[:, 1]
    return y_prob


def predict_all_horizons_rf(
    fitted_models: dict, df: pd.DataFrame, feature_cols: list[str]
):
    prediction_dict = {}
    for each_key in fitted_models.keys():
        each_model = fitted_models[each_key]["model"]
        each_imputer = fitted_models[each_key]["imputer"]
        y_prob = predict_random_forest_model(
            each_model, each_imputer, df, feature_cols)
        prediction_dict[each_key] = y_prob
    return prediction_dict


# Bonus function that prints which features matter most per horizon.
# Useful for deciding whether to add or drop features next iteration.
# It sorts by y_24 by default since
# that's usually the most informative middle horizon.
def get_feature_importances(
    fitted_models: dict, feature_cols: list[str]
) -> pd.DataFrame:
    """Returns a DataFrame of feature importances for each horizon."""
    rows = {}
    for key, val in fitted_models.items():
        rows[key] = val["model"].feature_importances_
    return pd.DataFrame(
        rows, index=feature_cols).sort_values("y_24", ascending=False)
