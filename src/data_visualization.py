import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import math
import os
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_target_balance(
        df: pd.DataFrame,
        target_cols: list[str],
        save_path: str | None = None,
        show: bool = True):
    positive_rates = [df[col].mean() for col in target_cols]
    labels = [col.replace("y_", "") + "h" for col in target_cols]
    fig, ax = plt.subplots()
    for i, v in enumerate(positive_rates):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    ax.bar(labels, positive_rates)
    ax.set_title("Positive Class Rate Across Horizon")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Positive Rate")
    ax.set_ylim(0, 0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list[str],
    hue_col: str | None = None,
    n_cols: int = 3,
    bins: int = 40,
    max_unique_for_bar: int = 12,
    bar_cols: list[str] | None = None,
    clip_quantile: float | None = 0.99,
    log_hist_cols: list[str] | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    Plot feature distributions:
    - bar chart for binary / low-cardinality discrete features
    - histogram for continuous features

    Extra options:
    - bar_cols: force selected columns to use bar charts
    - clip_quantile: clip extreme values for histogram plotting only
    - log_hist_cols: use log1p for selected nonnegative histogram columns
    """
    bar_cols = bar_cols or []
    log_hist_cols = log_hist_cols or []

    n_plots = len(feature_cols)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.8)
    )
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        series = df[col].dropna()
        n_unique = series.nunique()

        use_bar = (col in bar_cols) or (n_unique <= max_unique_for_bar)

        if use_bar:
            if hue_col and hue_col in df.columns:
                value_table = (
                    df[[col, hue_col]]
                    .dropna()
                    .groupby([col, hue_col])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )
                value_table.plot(kind="bar", ax=ax)
                ax.set_ylabel("Count")
                ax.legend(title=hue_col, fontsize=8)
            else:
                counts = series.value_counts().sort_index()
                ax.bar(counts.index.astype(str), counts.values)
                ax.set_ylabel("Count")

            ax.set_xlabel("")
            ax.set_title(col, fontsize=10)

        else:
            if hue_col and hue_col in df.columns:
                for label, group in df.groupby(hue_col):
                    plot_values = group[col].dropna()

                    if clip_quantile is not None:
                        upper = plot_values.quantile(clip_quantile)
                        lower = plot_values.quantile(
                            1 - clip_quantile
                            ) if plot_values.min() < 0 else None

                        if lower is not None:
                            plot_values = plot_values[
                                (plot_values >= lower) &
                                (plot_values <= upper)]
                        else:
                            plot_values = plot_values[plot_values <= upper]

                    if col in log_hist_cols:
                        plot_values = plot_values[plot_values >= 0]
                        plot_values = np.log1p(plot_values)
                        xlabel = f"log1p({col})"
                    else:
                        xlabel = "Value"

                    ax.hist(
                        plot_values,
                        bins=bins,
                        alpha=0.6,
                        label=f"{hue_col}={label}",
                        density=True,
                    )

                ax.set_ylabel("Density")
                ax.set_xlabel(xlabel)
                ax.legend(fontsize=8)
                ax.set_title(col, fontsize=10)

            else:
                plot_values = series.copy()

                if clip_quantile is not None:
                    upper = plot_values.quantile(clip_quantile)
                    lower = plot_values.quantile(
                        1 - clip_quantile
                        ) if plot_values.min() < 0 else None

                    if lower is not None:
                        plot_values = plot_values[
                            (plot_values >= lower) & (plot_values <= upper)]
                    else:
                        plot_values = plot_values[plot_values <= upper]

                if col in log_hist_cols:
                    plot_values = plot_values[plot_values >= 0]
                    plot_values = np.log1p(plot_values)
                    xlabel = f"log1p({col})"
                else:
                    xlabel = "Value"

                ax.hist(plot_values, bins=bins, alpha=0.8)
                ax.set_ylabel("Count")
                ax.set_xlabel(xlabel)
                ax.set_title(col, fontsize=10)

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_correlation_heatmap(
        df: pd.DataFrame,
        feature_cols: list[str],
        title: str,
        save_path: str | None = None,
        show: bool = True):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[feature_cols].corr()
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_boxplots_by_horizon(
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        save_path: str | None = None,
        show: bool = True):
    num_plots = len(feature_cols)
    ncols = 3
    nrows = math.ceil(num_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    if num_plots == 1:
        axes = [axes]
    for i, feature in enumerate(feature_cols):
        group0 = df[df[target_col] == 0][feature].dropna()
        group1 = df[df[target_col] == 1][feature].dropna()

        axes[i].boxplot([group0, group1], labels=["0", "1"], showfliers=False)
        axes[i].set_title(feature)
        axes[i].set_xlabel(target_col)
        axes[i].set_ylabel("Value")
    fig.suptitle("Feature Distributions by y_24", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_compare_models_by_horizon(
        model_1_evaluation: dict,
        model_2_evaluation: dict,
        model_1_name: str,
        model_2_name: str,
        key_word: str,
        save_path: str | None = None,
        show: bool = True):
    horizons = ["y_12", "y_24", "y_48", "y_72"]
    labels = ["12h", "24h", "48h", "72h"]

    model_1_values = [model_1_evaluation[h][key_word] for h in horizons]
    model_2_values = [model_2_evaluation[h][key_word] for h in horizons]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(labels, model_1_values, marker="o", label=model_1_name)
    ax.plot(labels, model_2_values, marker="o", label=model_2_name)

    ax.set_title(f"{key_word.upper()} by Prediction Horizon")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel(key_word.upper())

    if key_word.lower() == "auc":
        ax.set_ylim(0.9, 1.0)

    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_compare_roc_curves(
        y_true,
        y_prob_1,
        y_prob_2,
        model_1_name,
        model_2_name,
        save_path=None,
        show=True):
    fpr_1, tpr_1, _ = roc_curve(y_true, y_prob_1)
    auc_1 = auc(fpr_1, tpr_1)

    fpr_2, tpr_2, _ = roc_curve(y_true, y_prob_2)
    auc_2 = auc(fpr_2, tpr_2)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_1, tpr_1, label=f"{model_1_name} (AUC = {auc_1:.3f})")
    ax.plot(fpr_2, tpr_2, label=f"{model_2_name} (AUC = {auc_2:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

    ax.set_title("ROC Curve Comparison (24h)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_random_forest_feature_importance(
        model,
        feature_cols: list[str],
        top_n: int = 10,
        save_path: str | None = None,
        show: bool = True
):
    """
    Plot top feature importances from a trained Random Forest model.

    Args:
        model: trained RandomForestClassifier
        feature_cols: list of feature names used to train the model
        top_n: number of top features to display
        save_path: optional file path to save the plot
        show: whether to display the plot
    """
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    })
    importance_df = importance_df.sort_values(
        by="importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.invert_yaxis()

    ax.set_title(f"Top {top_n} Random Forest Feature Importances (24h)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig) 
    
    
