import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy.stats import wilcoxon


def prepare_dfs(
    df: pd.DataFrame, text_feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    female_df = df[df["gender"] == "female"][text_feature_cols].astype("float")
    male_df = df[df["gender"] == "male"][text_feature_cols].astype("float")

    return female_df, male_df


def compute_wilcoxon_pvalues(
    female_df: pd.DataFrame, male_df: pd.DataFrame, text_feature_cols: List[str]
) -> dict:
    wilcoxon_pvalues = dict()

    for feature in text_feature_cols:
        res = wilcoxon(
            np.array(female_df[feature]),
            np.array(male_df[feature]),
            alternative="two-sided",
        )
        wilcoxon_pvalues[feature] = res.pvalue

    wilcoxon_pvalues = {
        feature: pvalue
        for feature, pvalue in sorted(
            wilcoxon_pvalues.items(), key=lambda item: item[1]
        )
    }

    return wilcoxon_pvalues


def compute_mean_differences(
    male_df: pd.DataFrame, female_df: pd.DataFrame, text_feature_cols: dict
) -> dict:
    mean_differences = {
        col: male_df[col].mean() - female_df[col].mean() for col in text_feature_cols
    }

    normalized_differences = {
        col: mean_differences[col]
        / (
            max(female_df[col].max(), male_df[col].max())
            - min(female_df[col].min(), male_df[col].min())
        )
        for col in mean_differences
    }

    # Sort by absolute value
    sorted_means = dict(
        sorted(
            {
                feature: value
                for feature, value in normalized_differences.items()
                if not pd.isna(value)
            }.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    )

    return sorted_means
