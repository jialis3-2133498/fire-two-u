import matplotlib.pyplot as plt
import pandas as pd


def plot_target_balance(df: pd.DataFrame, target_cols: list[str]):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, each_target_col in enumerate(target_cols):
        counts = df[each_target_col].value_counts().sort_index()

        axes[i].pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[i].set_title(each_target_col)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_feature_histograms(
    df: pd.DataFrame,
    feature_cols: list[str],
    hue_col: str | None = None,
    n_cols: int = 3,
    bins: int = 40,
):
    """
    Plot histograms for each feature in feature_cols.

    Args:
        df:           DataFrame containing the features (e.g. train_split).
        feature_cols: List of column names to plot.
        hue_col:      Optional column to color-split each histogram
                      (e.g. 'event' to compare event=0 vs event=1).
        n_cols:       Number of subplot columns per row.
        bins:         Number of histogram bins.
    """
    n_rows = -(-len(feature_cols) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]

        if hue_col and hue_col in df.columns:
            # Overlay one histogram per class value
            for label, group in df.groupby(hue_col):
                ax.hist(
                    group[col].dropna(),
                    bins=bins,
                    alpha=0.6,
                    label=f"{hue_col}={label}",
                    density=True,  # normalize so different class sizes are comparable
                )
            ax.legend(fontsize=8)
        else:
            ax.hist(df[col].dropna(), bins=bins, color="steelblue", alpha=0.8)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density" if hue_col else "Count")

    # Hide any unused subplot slots
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Histograms", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str]):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     corr = df[feature_cols].corr()
#     im = ax.imshow(corr.to_numpy(), cmap="coolwarm")
#     ax.set_xticks(range(len(corr.columns)))
#     ax.set_xticklabels(corr.columns, rotation=45, ha="right")
#     ax.set_yticks()


def plot_auc_by_horizon(model_evaluation: dict):
    pass
