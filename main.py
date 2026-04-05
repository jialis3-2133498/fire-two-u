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
    plot_feature_distributions,
    plot_boxplots_by_horizon,
    plot_correlation_heatmap,
    plot_compare_models_by_horizon,
    plot_compare_roc_curves,
    plot_random_forest_feature_importance
)
from src.save_output import (
    save_feature_summary_table,
)


def main():

    # ──────────────── Data Cleaning ────────────────
    train_df = pd.read_csv("data/train_split.csv")
    valid_df = pd.read_csv("data/val_split.csv")
    test_df = pd.read_csv("data/test_clean.csv")

    # ──────────────── Create Labels ────────────────
    train_df = create_horizon_label(train_df)
    valid_df = create_horizon_label(valid_df)

    # ──────────────── Feature Selection ────────────────
    feature_cols = [
        "num_perimeters_0_5h",  # numerical
        "low_temporal_resolution_0_5h",  # numerical
        "log1p_area_first",  # numerical
        "log_area_ratio_0_5h",  # numerical
        "centroid_speed_m_per_h",  # numerical
        "spread_bearing_sin",  # numerical
        "spread_bearing_cos",  # numerical
        "dist_min_ci_0_5h",  # numerical
        "closing_speed_m_per_h",  # numerical
        "alignment_cos",  # numerical
        "event_start_hour",  # numerical
        "event_start_month",  # numerical
    ]
    target_cols = ["y_12", "y_24", "y_48", "y_72"]
    cols_for_heatmap = feature_cols + target_cols

    # ──────────────── Exploratory DA ────────────────
    summary_df = save_feature_summary_table(
        train_df,
        feature_cols,
        output_path="outputs/feature_summary.csv")
    print(summary_df)
    plot_target_balance(
        train_df,
        target_cols,
        save_path="outputs/figures/target_positive_rate.png",
        show=False)
    plot_feature_distributions(
        train_df,
        feature_cols,
        hue_col=None,
        n_cols=3,
        bins=40,
        max_unique_for_bar=12,
        save_path="outputs/figures/feature_histograms.png",
        show=False
    )
    plot_correlation_heatmap(
        train_df,
        feature_cols,
        title="Features Correlation Heatmap",
        save_path="outputs/figures/feature_correlation_heatmap.png",
        show=False
    )
    plot_correlation_heatmap(
        train_df,
        cols_for_heatmap,
        title="Feature and Target Correlation Heatmap",
        save_path="outputs/figures/feature_target_heatmap.png",
        show=False
    )

    cols_for_boxplot = [
        "num_perimeters_0_5h",
        "log1p_area_first",
        "dist_min_ci_0_5h",
    ]
    plot_boxplots_by_horizon(
        train_df,
        cols_for_boxplot,
        "y_24",
        save_path="outputs/figures/boxplot_y24.png",
        show=False)

    # ──────────────── LOGISTIC REGRESSION ────────────────
    print("\n=== Logistic Regression ===")
    # Train models
    lr_models = train_all_logistic_models(
        train_df,
        valid_df,
        feature_cols,
        target_cols)
    # Predict
    lr_predictions = predict_all_horizons(lr_models, valid_df, feature_cols)
    # Evaluate
    lr_model_evaluation = {}
    for each_key in lr_predictions.keys():
        y_true = valid_df[each_key]
        y_prob = lr_predictions[each_key]

        auc, report = evaluate_predictions(y_true, y_prob)
        lr_model_evaluation[each_key] = {"auc": auc, "report": report}

        print(f"\n=== {each_key} ===")
        print("AUC:", auc)
        print(report)

    # ──────────────── RANDOM FOREST ────────────────
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

    # ──────────────── FEATURE IMPORTANCES ────────────────
    # print("\n=== Feature Importances (RF) ===")
    # importances = get_feature_importances(rf_models, feature_cols)
    # print(importances.to_string())

    # 7. Predictions on test file
    lr_test_preds = predict_all_horizons(lr_models, test_df, feature_cols)
    rf_test_preds = predict_all_horizons_rf(rf_models, test_df, feature_cols)

    # 8. Acuraccy Plot by models
    plot_compare_models_by_horizon(
        lr_models,
        rf_models,
        "Logistic Model",
        "Random Forest",
        "auc",
        save_path="outputs/figures/model_compare_auc.png",
        show=False)
    
    y_true = valid_df["y_24"]
    y_prob_lr = lr_predictions["y_24"]
    y_prob_rf = rf_predictions["y_24"]

    plot_compare_roc_curves(
        y_true=y_true,
        y_prob_1=y_prob_lr,
        y_prob_2=y_prob_rf,
        model_1_name="Logistic Regression",
        model_2_name="Random Forest",
        save_path="outputs/figures/roc_compare_y24.png",
        show=True
    )
    plot_random_forest_feature_importance(
        model=rf_models["y_24"]["model"],
        feature_cols=feature_cols,
        top_n=10,
        save_path="outputs/figures/rf_feature_importance_y24.png",
        show=True
    )

    return (
        lr_model_evaluation,
        rf_model_evaluation,
        lr_test_preds,
        rf_test_preds)


if __name__ == "__main__":
    main()
