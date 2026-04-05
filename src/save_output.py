import pandas as pd
import matplotlib.pyplot as plt


def save_feature_summary_table(
        df: pd.DataFrame,
        feature_cols: list[str],
        output_path: str):
    summary = pd.DataFrame({
        "feature": feature_cols,
        "dtype": [df[col].dtype for col in feature_cols],
        "missing_count": [df[col].isna().sum() for col in feature_cols],
        "missing_pct": [df[col].isna().mean() for col in feature_cols],
        "unique_count": [df[col].nunique(dropna=True) for col in feature_cols],
        "mean": [df[col].mean() for col in feature_cols],
        "std": [df[col].std() for col in feature_cols],
        "min": [df[col].min() for col in feature_cols],
        "median": [df[col].median() for col in feature_cols],
        "max": [df[col].max() for col in feature_cols],
    })

    summary.to_csv(output_path, index=False)
    return summary


def save_feature_summary_table_image(
        summary_df: pd.DataFrame,
        output_path: str):
    fig, ax = plt.subplots(figsize=(14, max(4, len(summary_df) * 0.5)))
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.round(3).values,
        colLabels=summary_df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
