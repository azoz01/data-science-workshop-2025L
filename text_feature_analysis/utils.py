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
    """ 
    returns sorted pvalues
    """
    wilcoxon_pvalues = dict()

    available_cols = get_available_cols(text_feature_cols, model_name, male_df, female_df)
    
    for feature in available_cols:
        res = wilcoxon(
            np.array(female_df[feature]),
            np.array(male_df[feature]),
            alternative="two-sided",
            nan_policy="omit"
        )
        wilcoxon_pvalues[feature] = res.pvalue

    wilcoxon_pvalues = {
        feature: pvalue
        for feature, pvalue in sorted(
            wilcoxon_pvalues.items(), key=lambda item: item[1]
        )
    }

    return wilcoxon_pvalues


def compute_cohens_d(
    male_df: pd.DataFrame, female_df: pd.DataFrame, text_feature_cols: dict, model_name: str = None
) -> dict:
    available_cols = [col for col in text_feature_cols if col in male_df.columns and col in female_df.columns]
    missing_cols = set(text_feature_cols) - set(available_cols)

    if missing_cols:
        print(f"Missing columns: {missing_cols} in {model_name if model_name else 'unknown'}")

    cohens_d = {}
    for col in available_cols:
        male_vals = male_df[col].dropna()
        female_vals = female_df[col].dropna()

        n1, n2 = len(male_vals), len(female_vals)
        s1, s2 = male_vals.std(ddof=1), female_vals.std(ddof=1)
        s_pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

        if s_pooled == 0:
            cohens_d[col] = 0
        else:
            d = (male_vals.mean() - female_vals.mean()) / s_pooled
            cohens_d[col] = d

    return cohens_d


def sort_dict_by_abs_value(unsorted_dict: dict) -> dict:
    sorted_dict = dict(
        sorted(
            {
                feature: value
                for feature, value in unsorted_dict.items()
                if not pd.isna(value)
            }.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    )

    return sorted_dict


def get_sorted_cohens_d(male_df: pd.DataFrame, female_df: pd.DataFrame, text_feature_cols: dict, model_name: str=None) -> dict:
    cohens_ds = compute_cohens_d(male_df, female_df, text_feature_cols, model_name)
    sorted_choens_ds = sort_dict_by_abs_value(cohens_ds)
    
    return sorted_choens_ds
