import pandas as pd
from src.logistic_models import (
    create_horizon_label,
    train_all_logistic_models,
    predict_all_horizons,
    evaluate_predictions
    )
from src.data_visualization import (
    plot_target_balance,
    plot_correlation_heatmap
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
        "event_start_month"
    ]
    # feature_cols = train_df[feature_cols].select_dtypes(
    #     include="number").columns.tolist()
    target_cols = ["y_12", "y_24", "y_48", "y_72"]
    # for col in ["y_12", "y_24", "y_48", "y_72"]:
    #     print(col, train_df[col].value_counts(dropna=False))
    cols_for_heatmap = feature_cols + target_cols
    plot_target_balance(train_df, target_cols)
    plot_correlation_heatmap(train_df, feature_cols)
    plot_correlation_heatmap(train_df, cols_for_heatmap)
    # 4. train models
    models = train_all_logistic_models(train_df, valid_df, feature_cols, target_cols)
    # 5. predict
    predictions = predict_all_horizons(models, valid_df, feature_cols)
    # 6. evaluate
    model_evaluation = {}
    for each_key in predictions.keys():
        y_prob = predictions[each_key]
        auc, report = evaluate_predictions(valid_df[each_key], y_prob)
        model_evaluation[each_key] = {"auc": auc, "report": report}

        print(f"\n=== {each_key} ===")
        print("AUC:", auc)
        print(report)
    # 7. Predictions on test file
    test_predictions = predict_all_horizons(models, test_df, feature_cols)
    return model_evaluation, test_predictions


if __name__ == "__main__":
    main()
