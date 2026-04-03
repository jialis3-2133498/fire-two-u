import pandas as pd
from src.logistic_models import (
    create_horizon_label,
    train_all_logistic_models,
    predict_all_horizons,
    evaluate_predictions,
)
from src.data_visualization import plot_target_balance
from src.data_visualization import plot_feature_histograms


def main():
    # 1. load data
    train_df = pd.read_csv("data/train_split.csv")
    valid_df = pd.read_csv("data/val_split.csv")
    test_df = pd.read_csv("data/test_clean.csv")
    # 2. create labels
    train_df = create_horizon_label(train_df)
    valid_df = create_horizon_label(valid_df)

    # 3. define feature_cols and target_cols
    feature_cols = [
        c
        for c in train_df.columns
        if c
        not in [
            "event_id",
            "event",
            "time_to_hit_hours",
            "y_12",
            "y_24",
            "y_48",
            "y_72",
        ]
    ]
    # feature_cols = train_df[feature_cols].select_dtypes(
    #     include="number").columns.tolist()
    target_cols = ["y_12", "y_24", "y_48", "y_72"]
    plot_target_balance(train_df, target_cols)
    plot_feature_histograms(train_df, feature_cols)
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
