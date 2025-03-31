from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Tuple

group_colors = {
    "gpt": ("blue", 1),
    "claude": ("orange", 2),
    "mixtral": ("green", 3),
    "mistral": ("green", 3),
    "qwen": ("red", 4),
    "llama": ("purple", 5)
}

def get_group_and_color(name: str) -> Tuple[int, str]:
    name = name.lower()
    for group, (color, order) in group_colors.items():
        if group in name:
            return order, color
    return 99, "gray"  # Default if no match


def plot_features_distribution(
    male_df: pd.DataFrame,
    female_df: pd.DataFrame,
    all_pvalues: dict,
    pvalue: float,
    model_name: str,
    test_name: str = "Wilcoxon",
    max_cols: int = 5,
    base_width: int = 3,
    base_height: int = 3,
    path_to_save: str = None,
    show: bool = True
) -> None:
    # dynamically caluclate number of rows and columns
    num_features = len(all_pvalues)
    num_cols = min(max_cols, num_features)
    num_rows = int(np.ceil(num_features / num_cols))

    # dynamically set figure size based on rows and columns
    fig_width = num_cols * base_width
    fig_height = num_rows * base_height
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    fig.suptitle(
        f"Distributions of text features across genders in {model_name}, \n"
        f"{test_name} test, p-value < {pvalue}"
    )

    # flattering axes for easy iteration
    axes = np.ravel(axes)

    # plot KDE
    for ax, col in zip(axes, all_pvalues.keys()):
        sns.kdeplot(
            female_df[col], label="female", fill=True, alpha=0.5, ax=ax, color="hotpink"
        )
        sns.kdeplot(
            male_df[col], label="male", fill=True, alpha=0.5, ax=ax, color="skyblue"
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(col)
        ax.legend()

    # hide unused plots
    for ax in axes[num_features:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # save and show
    if path_to_save:
        plt.savefig(path_to_save+f'/{model_name}.png')
        
    plt.show() if show else plt.close()


def plot_mean_differences(sorted_means: dict, model_name: str, path_to_save: str = None, show: bool = True, across_models: bool = False) -> None:

    colors = ["skyblue" if val > 0 else "hotpink" for val in sorted_means.values()]

    # figure and plot
    figsize = (4, 6) if across_models else (12, 18)
    
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        y=list(sorted_means.keys()),
        x=list(sorted_means.values()),
        hue=list(sorted_means.keys()),
        palette=colors,
        ax=ax,
    )
    ax.set_xlabel("Normalized Mean Difference (male - female)")
    ax.set_ylabel("Columns")
    ax.grid()

    # legend and title
    legend_patches = [
        mpatches.Patch(color="hotpink", label="Female-dominated"),
        mpatches.Patch(color="skyblue", label="Male-dominated"),
    ]

    ax.legend(
        handles=legend_patches,
        loc="lower left",
        mode="expand",
        borderaxespad=0.0,
        ncol=2,
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.102),
    )
    ax.set_title(f"Normalized Differences Between Means in {model_name}", pad=40)

    # save and show
    if path_to_save:
        plt.savefig(path_to_save+f'/{model_name}.png')

    plt.show() if show else plt.close()


def plot_total_means_differences(total_mean_diff: dict) -> None:
    
    models = list(total_mean_diff.keys())
    values = list(total_mean_diff.values())
    
    model_data = [(model, values[i], *get_group_and_color(model)) for i, model in enumerate(models)]
    model_data.sort(key=lambda x: x[2])  

    sorted_models = [m[0] for m in model_data]
    sorted_values = [m[1] for m in model_data]
    sorted_colors = [m[3] for m in model_data]
    
    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=sorted_models, y=sorted_values, palette=sorted_colors)

    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Model Means Differences Comparison", fontsize=16)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.ylim(0, max(sorted_values) * 1.1)  

    plt.subplots_adjust(bottom=0.2, left=0.15, right=0.9)

    plt.show()