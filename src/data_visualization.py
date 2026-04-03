import matplotlib.pyplot as plt
import pandas as pd


def plot_target_balance(
        df: pd.DataFrame,
        target_cols: list[str]):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, each_target_col in enumerate(target_cols):
        counts = df[each_target_col].value_counts().sort_index()

        axes[i].pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=90
        )
        axes[i].set_title(each_target_col)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_feature_histograms(df: pd.DataFrame, feature_cols: list[str]):
    pass


# def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str]):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     corr = df[feature_cols].corr()
#     im = ax.imshow(corr.to_numpy(), cmap="coolwarm")
#     ax.set_xticks(range(len(corr.columns)))
#     ax.set_xticklabels(corr.columns, rotation=45, ha="right")
#     ax.set_yticks()


def plot_auc_by_horizon(model_evaluation: dict):
    pass
