import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy.stats import wilcoxon


def prepare_dfs(
    df: pd.DataFrame, text_feature_cols: List[str], df_name: str=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    available_cols = [col for col in text_feature_cols if col in df.columns]
    missing_cols = set(text_feature_cols) - set(available_cols)

    if missing_cols:
        print(f"Missing columns: {missing_cols} in {df_name if df_name else 'unknown'}")  

    female_df = df[df["gender"] == "female"][available_cols].astype("float")
    male_df = df[df["gender"] == "male"][available_cols].astype("float")

    return female_df, male_df


def get_available_cols(text_feature_cols: List[str], model_name: str, male_df: pd.DataFrame, female_df: pd.DataFrame):
    available_cols = [col for col in text_feature_cols if col in male_df.columns and col in female_df.columns]
    missing_cols = set(text_feature_cols) - set(available_cols)

    if missing_cols:
        print(f"Missing columns: {missing_cols} in {model_name if model_name else 'unknown'}")  

    return available_cols

def compute_wilcoxon_pvalues(
    female_df: pd.DataFrame, male_df: pd.DataFrame, text_feature_cols: List[str], model_name: str
) -> dict:
    wilcoxon_pvalues = dict()

    available_cols = get_available_cols(text_feature_cols, model_name, male_df, female_df)
    
    for feature in available_cols:
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
    male_df: pd.DataFrame, female_df: pd.DataFrame, text_feature_cols: dict, model_name: str=None
) -> dict:
    available_cols = [col for col in text_feature_cols if col in male_df.columns and col in female_df.columns]
    missing_cols = set(text_feature_cols) - set(available_cols)

    if missing_cols:
        print(f"Missing columns: {missing_cols} in {model_name if model_name else 'unknown'}")  

    mean_differences = {
        col: male_df[col].mean() - female_df[col].mean() for col in available_cols
    }
    
    normalized_differences = {
        col: mean_differences[col]
        / (
            max(female_df[col].max(), male_df[col].max())
            - min(female_df[col].min(), male_df[col].min())
        )
        for col in mean_differences
    }

    return normalized_differences


def sort_dict_by_abs_value(mean_differences: dict) -> dict:
    sorted_means = dict(
        sorted(
            {
                feature: value
                for feature, value in mean_differences.items()
                if not pd.isna(value)
            }.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    )

    return sorted_means


def get_sorted_mean_differences(male_df: pd.DataFrame, female_df: pd.DataFrame, text_feature_cols: dict, model_name: str=None) -> dict:
    mean_differences = compute_mean_differences(male_df, female_df, text_feature_cols, model_name)
    sorted_differences = sort_dict_by_abs_value(mean_differences)
    
    return sorted_differences