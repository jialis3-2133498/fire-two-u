import pandas as pd
from src.logistic_models import (
    create_horizon_label,
    train_all_logistic_models,
    predict_all_horizons,
    evaluate_predictions,
)
from src.random_forest_models import (
    train_all_random_forest_models,
    predict_all_horizons_rf,
    # get_feature_importances,
)
from src.data_visualization import (
    plot_target_balance,
    plot_feature_histograms,
    plot_correlation_heatmap,
)


def main():
    # 1. load data
    train_df = pd.read_csv("data/train_split.csv")
    valid_df = pd.read_csv("data/val_split.csv")
    test_df = pd.read_csv("data/test_clean.csv")
    # print(train_df.dtypes)
    # print(train_df.nunique().sort_values())

    # 2. create labels
    train_df = create_horizon_label(train_df)
    valid_df = create_horizon_label(valid_df)

    # 3. define feature_cols and target_cols
    feature_cols = [
        "num_perimeters_0_5h",
        "low_temporal_resolution_0_5h",
        "log1p_area_first",
        "log_area_ratio_0_5h",
        "centroid_speed_m_per_h",
        "spread_bearing_sin",
        "spread_bearing_cos",
        "dist_min_ci_0_5h",
        "closing_speed_m_per_h",
        "alignment_cos",
        "event_start_hour",
        "event_start_month",
    ]
    # feature_cols = train_df[feature_cols].select_dtypes(
    #     include="number").columns.tolist()
    target_cols = ["y_12", "y_24", "y_48", "y_72"]
    # for col in ["y_12", "y_24", "y_48", "y_72"]:
    #     print(col, train_df[col].value_counts(dropna=False))
    cols_for_heatmap = feature_cols + target_cols
    plot_target_balance(train_df, target_cols)
    plot_feature_histograms(train_df, feature_cols)
    plot_correlation_heatmap(train_df, feature_cols)
    plot_correlation_heatmap(train_df, cols_for_heatmap)

    # ── LOGISTIC REGRESSION ───────────────────────────────────────────────────
    print("\n=== Logistic Regression ===")
    # 4. train models
    lr_models = train_all_logistic_models(train_df, valid_df, feature_cols, target_cols)
    # 5. predict
    lr_predictions = predict_all_horizons(lr_models, valid_df, feature_cols)
    # 6. evaluate
    lr_model_evaluation = {}
    for each_key in lr_predictions.keys():
        y_prob = lr_predictions[each_key]
        auc, report = evaluate_predictions(valid_df[each_key], y_prob)
        lr_model_evaluation[each_key] = {"auc": auc, "report": report}

        print(f"\n=== {each_key} ===")
        print("AUC:", auc)
        print(report)

    # ── RANDOM FOREST ─────────────────────────────────────────────────────────
    print("\n=== Random Forest ===")
    rf_models = train_all_random_forest_models(
        train_df, valid_df, feature_cols, target_cols
    )
    rf_predictions = predict_all_horizons_rf(rf_models, valid_df, feature_cols)

    rf_model_evaluation = {}
    for key, y_prob in rf_predictions.items():
        auc, report = evaluate_predictions(valid_df[key], y_prob)
        rf_model_evaluation[key] = {"auc": auc, "report": report}
        print(f"\n  [{key}] AUC: {auc:.4f}")
        print(report)

    # ── FEATURE IMPORTANCES ───────────────────────────────────────────────────
    # print("\n=== Feature Importances (RF) ===")
    # importances = get_feature_importances(rf_models, feature_cols)
    # print(importances.to_string())

    # 7. Predictions on test file
    lr_test_preds = predict_all_horizons(lr_models, test_df, feature_cols)
    rf_test_preds = predict_all_horizons_rf(rf_models, test_df, feature_cols)

    return lr_model_evaluation, rf_model_evaluation, lr_test_preds, rf_test_preds


if __name__ == "__main__":
    main()
