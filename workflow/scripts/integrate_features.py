#!/usr/bin/env python3
"""
Integrate TE and ScanFold2 Features
====================================
Combine TE features with ScanFold2 structural features for joint analysis.
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegratedFeatureAnalyzer:
    """Analyze combined TE and ScanFold2 features."""

    def __init__(
        self, te_features_file: str, scanfold_features_file: str, output_prefix: str
    ):
        self.te_features_file = te_features_file
        self.scanfold_features_file = scanfold_features_file
        self.output_prefix = output_prefix

        self.te_features = None
        self.scanfold_features = None
        self.combined_features = None

    def load_data(self):
        """Load TE and ScanFold2 feature files."""
        logger.info("Loading feature data...")

        # Load TE features
        self.te_features = pd.read_csv(self.te_features_file)
        logger.info(f"Loaded {len(self.te_features)} transcripts with TE features")

        # Load ScanFold2 features
        self.scanfold_features = pd.read_csv(self.scanfold_features_file)
        logger.info(
            f"Loaded {len(self.scanfold_features)} transcripts with ScanFold2 features"
        )

    def merge_features(self):
        """Merge TE and ScanFold2 features on transcript ID."""
        logger.info("Merging feature sets...")

        # Merge on transcript_id
        self.combined_features = self.te_features.merge(
            self.scanfold_features,
            on="transcript_id",
            how="inner",
            suffixes=("_te", "_sf"),
        )

        logger.info(f"Combined dataset: {len(self.combined_features)} transcripts")
        logger.info(f"Total features: {len(self.combined_features.columns)}")

        # Save combined features
        output_file = f"{self.output_prefix}_combined_features.csv"
        self.combined_features.to_csv(output_file, index=False)
        logger.info(f"Saved combined features to {output_file}")

        return self.combined_features

    def analyze_feature_correlations(self):
        """Analyze correlations between TE and structural features."""
        logger.info("Analyzing TE-structure correlations...")

        # Select key features for correlation analysis
        te_cols = [
            "te_coverage_pct",
            "te_count",
            "line_count",
            "sine_count",
            "mean_te_divergence",
            "young_te_count",
        ]

        sf_cols = [
            "min_zscore",
            "mean_zscore",
            "zscore_below_minus1_fraction",
            "mean_mfe",
            "mean_ed",
            "base_pair_density",
        ]

        # Filter columns that exist
        te_cols = [c for c in te_cols if c in self.combined_features.columns]
        sf_cols = [c for c in sf_cols if c in self.combined_features.columns]

        if not te_cols or not sf_cols:
            logger.warning(
                "Not all expected columns found. Skipping correlation analysis."
            )
            return None

        # Calculate correlations
        correlations = []

        for te_col in te_cols:
            for sf_col in sf_cols:
                # Remove NaN values
                valid_data = self.combined_features[[te_col, sf_col]].dropna()

                if len(valid_data) > 10:
                    # Spearman correlation (non-parametric)
                    corr, pval = stats.spearmanr(valid_data[te_col], valid_data[sf_col])

                    correlations.append(
                        {
                            "te_feature": te_col,
                            "structure_feature": sf_col,
                            "correlation": corr,
                            "p_value": pval,
                        }
                    )

        corr_df = pd.DataFrame(correlations)

        # Save correlations
        output_file = f"{self.output_prefix}_te_structure_correlations.csv"
        corr_df.to_csv(output_file, index=False)
        logger.info(f"Saved correlations to {output_file}")

        # Report significant correlations
        sig_corr = corr_df[corr_df["p_value"] < 0.05].sort_values(
            "correlation", key=abs, ascending=False
        )

        logger.info(f"Found {len(sig_corr)} significant correlations (p < 0.05)")
        logger.info("\nTop 5 correlations:")
        for idx, row in sig_corr.head().iterrows():
            logger.info(
                f"  {row['te_feature']} vs {row['structure_feature']}: "
                f"r={row['correlation']:.3f}, p={row['p_value']:.2e}"
            )

        return corr_df

    def perform_integrated_pca(self):
        """Perform PCA on combined feature set."""
        logger.info("Performing integrated PCA...")

        # Get numeric columns excluding IDs and metadata
        exclude_cols = [
            "transcript_id",
            "chrom",
            "transcript_type",
            "group",
            "length",
            "sample",
        ]

        numeric_cols = self.combined_features.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        # Prepare data
        X = self.combined_features[feature_cols].fillna(0)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=min(10, len(feature_cols)))
        X_pca = pca.fit_transform(X_scaled)

        # Save PCA results
        pca_df = pd.DataFrame(
            X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
        )
        pca_df["transcript_id"] = self.combined_features["transcript_id"].values

        if "group" in self.combined_features.columns:
            pca_df["group"] = self.combined_features["group"].values
        elif "transcript_type" in self.combined_features.columns:
            # Create group from transcript_type
            coding_types = ["protein_coding", "mRNA"]
            lncrna_types = ["lncRNA", "lincRNA", "antisense"]

            pca_df["group"] = "Other"
            mask_coding = self.combined_features["transcript_type"].isin(coding_types)
            mask_lncrna = self.combined_features["transcript_type"].isin(lncrna_types)
            pca_df.loc[mask_coding, "group"] = "Coding"
            pca_df.loc[mask_lncrna, "group"] = "lncRNA"

        output_file = f"{self.output_prefix}_integrated_pca_scores.csv"
        pca_df.to_csv(output_file, index=False)
        logger.info(f"Saved PCA scores to {output_file}")

        # Save loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=feature_cols,
        )

        loadings_file = f"{self.output_prefix}_integrated_pca_loadings.csv"
        loadings.to_csv(loadings_file)

        # Report variance explained
        logger.info(
            f"PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance"
        )
        logger.info(
            f"First 5 PCs explain {np.sum(pca.explained_variance_ratio_[:5])*100:.2f}% of variance"
        )

        # Identify top contributing features to PC1
        pc1_loadings = loadings["PC1"].abs().sort_values(ascending=False)
        logger.info("\nTop 10 features contributing to PC1:")
        for feat, loading in pc1_loadings.head(10).items():
            logger.info(f"  {feat}: {loading:.3f}")

        return pca_df, loadings

    def perform_integrated_classification(self):
        """Train classifier on combined features."""
        logger.info("Training integrated classifier...")

        # Prepare data
        exclude_cols = [
            "transcript_id",
            "chrom",
            "transcript_type",
            "group",
            "length",
            "sample",
        ]

        numeric_cols = self.combined_features.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        # Define groups
        if "group" in self.combined_features.columns:
            analysis_data = self.combined_features[
                self.combined_features["group"].isin(["Coding", "lncRNA"])
            ].copy()
            y = (analysis_data["group"] == "lncRNA").astype(int)
        elif "transcript_type" in self.combined_features.columns:
            coding_types = ["protein_coding", "mRNA"]
            lncrna_types = ["lncRNA", "lincRNA", "antisense"]

            analysis_data = self.combined_features[
                self.combined_features["transcript_type"].isin(
                    coding_types + lncrna_types
                )
            ].copy()

            y = analysis_data["transcript_type"].isin(lncrna_types).astype(int)
        else:
            logger.warning("No group column found. Skipping classification.")
            return None

        X = analysis_data[feature_cols].fillna(0)

        # Train Random Forest with cross-validation
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        # Stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Get multiple metrics
        metrics = {}
        for metric in ["roc_auc", "accuracy", "f1"]:
            scores = cross_val_score(rf, X, y, cv=cv, scoring=metric)
            metrics[metric] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "scores": scores,
            }
            logger.info(f"{metric.upper()}: {scores.mean():.3f} ± {scores.std():.3f}")

        # Train on full data for feature importance
        rf.fit(X, y)

        # Feature importance
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Categorize features
        importance_df["category"] = importance_df["feature"].apply(
            lambda x: (
                "TE"
                if any(
                    te_word in x.lower()
                    for te_word in [
                        "te_",
                        "line",
                        "sine",
                        "ltr",
                        "dna",
                        "herv",
                        "alu",
                        "divergence",
                    ]
                )
                else "Structure"
            )
        )

        output_file = f"{self.output_prefix}_integrated_feature_importance.csv"
        importance_df.to_csv(output_file, index=False)
        logger.info(f"Saved feature importance to {output_file}")

        # Report top features by category
        logger.info("\nTop 5 TE features:")
        te_feats = importance_df[importance_df["category"] == "TE"].head()
        for idx, row in te_feats.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info("\nTop 5 Structure features:")
        struct_feats = importance_df[importance_df["category"] == "Structure"].head()
        for idx, row in struct_feats.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df, metrics

    def generate_integrated_visualizations(self):
        """Create visualizations for integrated analysis."""
        logger.info("Generating integrated visualizations...")

        # Load PCA scores if available
        pca_file = f"{self.output_prefix}_integrated_pca_scores.csv"
        try:
            pca_df = pd.read_csv(pca_file)

            # PCA plot
            fig, ax = plt.subplots(figsize=(10, 8))

            for group, color in [("Coding", "steelblue"), ("lncRNA", "coral")]:
                group_data = pca_df[pca_df["group"] == group]
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
            ax.set_title(
                "Integrated PCA: TE + Structure Features",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = f"{self.output_prefix}_integrated_pca_plot.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved PCA plot to {output_file}")

        except FileNotFoundError:
            logger.warning("PCA scores file not found. Skipping PCA plot.")

    def run_full_analysis(self):
        """Run complete integrated analysis."""
        self.load_data()
        self.merge_features()
        self.analyze_feature_correlations()
        self.perform_integrated_pca()
        self.perform_integrated_classification()
        self.generate_integrated_visualizations()

        logger.info("Integrated analysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate TE and ScanFold2 features for joint analysis"
    )
    parser.add_argument("--te-features", required=True, help="TE features CSV file")
    parser.add_argument(
        "--scanfold-features", required=True, help="ScanFold2 features CSV file"
    )
    parser.add_argument("--output-prefix", required=True, help="Output file prefix")

    args = parser.parse_args()

    analyzer = IntegratedFeatureAnalyzer(
        te_features_file=args.te_features,
        scanfold_features_file=args.scanfold_features,
        output_prefix=args.output_prefix,
    )

    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
