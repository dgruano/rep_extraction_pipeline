#!/usr/bin/env python3
"""
TE Feature Visualization
=========================
Generate comprehensive visualizations for TE feature analysis.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
sns.set_palette("Set2")


class TEVisualizer:
    """Generate visualizations for TE analysis results."""

    def __init__(
        self,
        features_file: str,
        test_results_file: str,
        pca_scores_file: str,
        output_dir: str,
    ):
        self.features_file = features_file
        self.test_results_file = test_results_file
        self.pca_scores_file = pca_scores_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.features = None
        self.test_results = None
        self.pca_scores = None

    def load_data(self):
        """Load all data files."""
        logger.info("Loading data for visualization...")

        self.features = pd.read_csv(self.features_file)

        # Calculate global presence flag
        self.features["hit_present"] = self.features["global_rm_count"] > 0

        if Path(self.test_results_file).exists():
            self.test_results = pd.read_csv(self.test_results_file)

        if Path(self.pca_scores_file).exists():
            self.pca_scores = pd.read_csv(self.pca_scores_file)

        logger.info("Data loaded successfully")

    def plot_hit_presence_comparison(self):
        """Plot TE presence comparison between groups."""
        logger.info("Plotting TE presence comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar plot of TE presence
        presence_data = (
            self.features[self.features["coding_class"].isin(["coding", "lncRNA"])]
            .groupby("coding_class")["hit_present"]
            .agg(["sum", "count"])
        )
        presence_data["percentage"] = (
            presence_data["sum"] / presence_data["count"]
        ) * 100

        axes[0].bar(
            presence_data.index,
            presence_data["percentage"],
            color=["steelblue", "coral"],
        )
        axes[0].set_ylabel("Percentage of transcripts with TEs (%)", fontsize=12)
        axes[0].set_title(
            "TE Presence by Transcript Type", fontsize=14, fontweight="bold"
        )
        axes[0].set_ylim([0, 100])

        # Add value labels
        for i, (idx, row) in enumerate(presence_data.iterrows()):
            axes[0].text(
                i,
                row["percentage"] + 2,
                f"{row['percentage']:.1f}%",
                ha="center",
                fontsize=11,
            )

        # TE count distribution
        hit_counts = self.features[
            (self.features["coding_class"].isin(["coding", "lncRNA"]))
            & (self.features["global_rm_count"] > 0)
        ]

        for group in ["coding", "lncRNA"]:
            group_data = hit_counts[hit_counts["coding_class"] == group][
                "global_rm_count"
            ]
            axes[1].hist(group_data, bins=30, alpha=0.6, label=group, density=True)

        axes[1].set_xlabel("Number of TE elements per transcript", fontsize=12)
        axes[1].set_ylabel("Density", fontsize=12)
        axes[1].set_title(
            "TE Element Count Distribution", fontsize=14, fontweight="bold"
        )
        axes[1].legend()
        axes[1].set_xlim([0, min(50, hit_counts["global_rm_count"].max())])

        plt.tight_layout()
        output_file = self.output_dir / "hit_presence_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_file}")

    def plot_coverage_comparison(self):
        """Plot TE coverage comparison."""
        logger.info("Plotting TE coverage comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        coverage_data = self.features[
            (self.features["coding_class"].isin(["coding", "lncRNA"]))
            & (self.features["global_rm_total_length_pct"] > 0)
        ]

        # Violin plot
        sns.violinplot(
            data=coverage_data,
            x="coding_class",
            y="global_rm_total_length_pct",
            ax=axes[0],
            cut=0,
        )
        axes[0].set_ylabel("TE coverage (%)", fontsize=12)
        axes[0].set_xlabel("")
        axes[0].set_title("TE Coverage Distribution", fontsize=14, fontweight="bold")
        axes[0].set_ylim(
            [0, min(100, coverage_data["global_rm_total_length_pct"].quantile(0.99))]
        )

        # Box plot with statistics
        bp = axes[1].boxplot(
            [
                coverage_data[coverage_data["coding_class"] == "coding"][
                    "global_rm_total_length_pct"
                ],
                coverage_data[coverage_data["coding_class"] == "lncRNA"][
                    "global_rm_total_length_pct"
                ],
            ],
            labels=["coding", "lncRNA"],
            patch_artist=True,
            showmeans=True,
        )

        # Color boxes
        for patch, color in zip(bp["boxes"], ["steelblue", "coral"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        axes[1].set_ylabel("TE coverage (%)", fontsize=12)
        axes[1].set_title("TE Coverage Box Plot", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "hit_coverage_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_file}")

    def plot_family_composition(self):
        """Plot TE family composition comparison."""
        logger.info("Plotting TE family composition...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Prepare data
        family_cols = ["te_line_count", "te_sine_count", "te_ltr_count", "te_dna_count"]

        for ax, group in zip(axes, ["coding", "lncRNA"]):
            group_data = self.features[self.features["coding_class"] == group][
                family_cols
            ].sum()

            if len(group_data) == 0:
                logger.warning(f"No valid data for {group} family composition")
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue
            # Pie chart
            colors = plt.cm.Set3(range(len(group_data)))
            wedges, texts, autotexts = ax.pie(
                group_data.values,
                labels=["LINE", "SINE", "LTR", "DNA"],
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )

            ax.set_title(
                f"{group} TE Family Composition", fontsize=14, fontweight="bold"
            )

            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

        plt.tight_layout()
        output_file = self.output_dir / "hit_family_composition.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_file}")

    def plot_volcano_plot(self):
        """Create volcano plot of statistical test results."""
        if self.test_results is None:
            logger.warning("No test results available for volcano plot")
            return

        logger.info("Creating volcano plot...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate -log10(p-value) and log2(fold change)
        data = self.test_results.copy()
        data["neg_log10_p"] = -np.log10(data["mann_whitney_p"] + 1e-300)
        data["log2_fc"] = np.log2(data["fold_change"] + 1e-10)

        # Significance thresholds
        sig_threshold = -np.log10(0.05)
        fc_threshold = 1  # 2-fold change

        # Color by significance
        data["color"] = "gray"
        data.loc[
            (data["neg_log10_p"] > sig_threshold) & (data["log2_fc"] > fc_threshold),
            "color",
        ] = "red"
        data.loc[
            (data["neg_log10_p"] > sig_threshold) & (data["log2_fc"] < -fc_threshold),
            "color",
        ] = "blue"

        # Scatter plot
        for color, label in [
            ("gray", "Not significant"),
            ("red", "Enriched in lncRNA"),
            ("blue", "Enriched in coding"),
        ]:
            subset = data[data["color"] == color]
            ax.scatter(
                subset["log2_fc"],
                subset["neg_log10_p"],
                c=color,
                alpha=0.6,
                s=50,
                label=label,
            )

        # Add threshold lines
        ax.axhline(y=sig_threshold, color="black", linestyle="--", alpha=0.5)
        ax.axvline(x=fc_threshold, color="black", linestyle="--", alpha=0.5)
        ax.axvline(x=-fc_threshold, color="black", linestyle="--", alpha=0.5)

        # Labels
        ax.set_xlabel("log2(Fold Change: lncRNA / coding)", fontsize=12)
        ax.set_ylabel("-log10(p-value)", fontsize=12)
        ax.set_title(
            "Volcano Plot: TE Feature Enrichment", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Annotate top features
        top_features = data.nlargest(5, "neg_log10_p")
        for _, row in top_features.iterrows():
            ax.annotate(
                row["feature"],
                xy=(row["log2_fc"], row["neg_log10_p"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        plt.tight_layout()
        output_file = self.output_dir / "volcano_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_file}")

    def plot_pca(self):
        """Plot PCA results."""
        if self.pca_scores is None:
            logger.warning("No PCA scores available")
            return

        logger.info("Plotting PCA...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        for group, color in [("coding", "steelblue"), ("lncRNA", "coral")]:
            group_data = self.pca_scores[self.pca_scores["coding_class"] == group]
            ax.scatter(
                group_data["PC1"],
                group_data["PC2"],
                c=color,
                label=group,
                alpha=0.6,
                s=50,
            )

        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.set_title("PCA: TE Features", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "pca_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_file}")

    def generate_all_plots(self):
        """Generate all visualizations."""
        self.load_data()
        self.plot_hit_presence_comparison()
        self.plot_coverage_comparison()
        self.plot_family_composition()
        self.plot_volcano_plot()
        self.plot_pca()

        logger.info("All visualizations complete!")


def main():
    parser = argparse.ArgumentParser(description="Generate TE analysis visualizations")
    parser.add_argument("--features", required=True, help="TE features CSV")
    parser.add_argument(
        "--test-results", required=True, help="Statistical test results CSV"
    )
    parser.add_argument("--pca-scores", required=True, help="PCA scores CSV")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for plots"
    )

    args = parser.parse_args()

    visualizer = TEVisualizer(
        features_file=args.features,
        test_results_file=args.test_results,
        pca_scores_file=args.pca_scores,
        output_dir=args.output_dir,
    )

    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
