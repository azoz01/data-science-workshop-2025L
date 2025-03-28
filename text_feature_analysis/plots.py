from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


def plot_features_distribution(
    male_df: pd.DataFrame,
    female_df: pd.DataFrame,
    all_pvalues: dict,
    pvalue: float,
    test_name: str = "Wilcoxon",
    max_cols: int = 5,
    base_width: int = 3,
    base_height: int = 3,
    path_to_save: str = None,
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
        "Distributions of text features across genders, \n"
        f"{test_name} test, p-value < {pvalue}",
        fontsize=16,
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
        plt.savefig(path_to_save)
    plt.show()


def plot_mean_differences(sorted_means: dict, path_to_save: str = None) -> None:

    colors = ["skyblue" if val > 0 else "hotpink" for val in sorted_means.values()]

    # figure and plot
    fig, ax = plt.subplots(figsize=(10, 18))
    sns.barplot(
        y=list(sorted_means.keys()),
        x=list(sorted_means.values()),
        hue=list(sorted_means.keys()),
        palette=colors,
        ax=ax,
    )
    ax.set_xlabel("Normalized Mean Difference (male - female)")
    ax.set_ylabel("Columns")
    ax.set_title("Normalized Differences Between Means")
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
    ax.set_title("Normalized Differences Between Means", pad=40)

    # save and show
    if path_to_save:
        plt.savefig()

    plt.show()
