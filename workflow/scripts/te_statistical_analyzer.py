#!/usr/bin/env python3
"""
TE Statistical Analyzer
=======================
Perform statistical comparisons of TE features between coding and lncRNA transcripts.
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, ks_2samp, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TEStatisticalAnalyzer:
    """Statistical analysis of TE features between transcript groups."""

    def __init__(self, features_file: str, output_prefix: str):
        self.features_file = features_file
        self.output_prefix = output_prefix

        self.all_data = None
        self.data = None
        self.coding_data = None
        self.lncrna_data = None
        self.results = {}

    def load_data(self):
        """Load feature matrix."""
        logger.info("Loading feature data...")

        self.all_data = pd.read_csv(self.features_file)
        self.data = self.all_data[
            self.all_data["coding_class"].isin(["coding", "lncRNA"])
        ].copy()
        logger.info(
            f"Loaded {len(self.data)} transcripts with {len(self.data.columns)} features"
        )

        # Calculate global presence flag
        self.data["hit_present"] = self.data["global_rm_count"] > 0

        self.coding_data = self.data[self.data["coding_class"] == "coding"]
        self.lncrna_data = self.data[self.data["coding_class"] == "lncRNA"]

        logger.info(f"Coding transcripts: {len(self.coding_data)}")
        logger.info(f"lncRNA transcripts: {len(self.lncrna_data)}")

    def get_numeric_features(self) -> list:
        """Get list of numeric feature columns."""
        exclude_cols = [
            "transcript_id",
            "chrom",
            "transcript_type",
            "coding_class",
            "transcript_length",
        ]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in exclude_cols]

    def perform_univariate_tests(self):
        """Perform univariate statistical tests for each feature."""
        logger.info("Performing univariate statistical tests...")

        features = self.get_numeric_features()
        test_results = []

        for feature in features:
            coding_vals = self.coding_data[feature].dropna()
            lncrna_vals = self.lncrna_data[feature].dropna()

            if len(coding_vals) == 0 or len(lncrna_vals) == 0:
                continue

            # Mann-Whitney U test
            statistic, p_value = mannwhitneyu(
                coding_vals, lncrna_vals, alternative="two-sided"
            )

            # Effect size (rank biserial correlation)
            n1, n2 = len(coding_vals), len(lncrna_vals)
            effect_size = 1 - (2 * statistic) / (n1 * n2)

            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = ks_2samp(coding_vals, lncrna_vals)

            test_results.append(
                {
                    "feature": feature,
                    "coding_mean": coding_vals.mean(),
                    "coding_median": coding_vals.median(),
                    "coding_std": coding_vals.std(),
                    "lncrna_mean": lncrna_vals.mean(),
                    "lncrna_median": lncrna_vals.median(),
                    "lncrna_std": lncrna_vals.std(),
                    "fold_change": lncrna_vals.mean() / (coding_vals.mean() + 1e-10),
                    "mann_whitney_u": statistic,
                    "mann_whitney_p": p_value,
                    "effect_size": effect_size,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_pval,
                }
            )

        results_df = pd.DataFrame(test_results)

        # Multiple testing correction (Benjamini-Hochberg)
        if len(results_df) > 0:
            _, results_df["mann_whitney_q"], _, _ = multipletests(
                results_df["mann_whitney_p"], method="fdr_bh"
            )
            _, results_df["ks_q_value"], _, _ = multipletests(
                results_df["ks_p_value"], method="fdr_bh"
            )

        # Sort by significance
        results_df = results_df.sort_values("mann_whitney_p")

        self.results["univariate_tests"] = results_df

        # Save results
        output_file = os.path.join(self.output_prefix, "univariate_tests.csv")
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved univariate test results to {output_file}")

        # Report significant features
        sig_features = results_df[results_df["mann_whitney_q"] < 0.05]
        logger.info(f"Found {len(sig_features)} significant features (FDR < 0.05)")

        return results_df

    def perform_categorical_tests(self):
        """Perform tests on categorical TE features."""
        logger.info("Performing categorical tests...")

        categorical_results = []

        contingency = pd.crosstab(self.data["coding_class"], self.data["hit_present"])

        if contingency.shape == (2, 2):
            logger.info("Performing Fisher's exact test...")
            odds_ratio, p_value = fisher_exact(contingency)

            logger.info(f"Odds ratio: {odds_ratio:.4f}, p-value: {p_value:.4e}")

            categorical_results.append(
                {
                    "test": "TE presence",
                    "coding_positive": (
                        contingency.loc["coding", 1] if 1 in contingency.columns else 0
                    ),
                    "coding_total": contingency.loc["coding"].sum(),
                    "lncrna_positive": (
                        contingency.loc["lncRNA", 1] if 1 in contingency.columns else 0
                    ),
                    "lncrna_total": contingency.loc["lncRNA"].sum(),
                    "odds_ratio": odds_ratio,
                    "p_value": p_value,
                }
            )
            logger.info("Fisher's exact test completed successfully")
        else:
            logger.warning(
                f"Contingency table shape {contingency.shape} is not 2x2, skipping Fisher's exact test"
            )

        results_df = pd.DataFrame(categorical_results)
        logger.info(f"Number of categorical test results: {len(results_df)}")

        if len(results_df) > 0:
            output_file = os.path.join(self.output_prefix, "categorical_tests.csv")
            results_df.to_csv(output_file, index=False)
            logger.info(f"Saved categorical test results to {output_file}")
        else:
            logger.warning("No categorical test results to save!")

        self.results["categorical_tests"] = results_df

        logger.info("CATEGORICAL TESTS COMPLETED")
        logger.info("=" * 60)

        return results_df

    def perform_pca(self):
        """Perform PCA on TE features."""
        logger.info("Performing PCA analysis...")

        features = self.get_numeric_features()

        # Prepare data (only transcripts with group labels)
        analysis_data = self.data[
            self.data["coding_class"].isin(["coding", "lncRNA"])
        ].copy()

        # Extract feature matrix
        X = analysis_data[features].fillna(0)
        y = analysis_data["coding_class"]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=min(10, len(features)))
        X_pca = pca.fit_transform(X_scaled)

        # Save PCA results
        pca_results = pd.DataFrame(
            X_pca[:, :5], columns=[f"PC{i+1}" for i in range(min(5, X_pca.shape[1]))]
        )
        pca_results["coding_class"] = y.values
        pca_results["transcript_id"] = analysis_data["transcript_id"].values

        output_file = os.path.join(self.output_prefix, "pca_scores.csv")
        pca_results.to_csv(output_file, index=False)
        logger.info(f"Saved PCA scores to {output_file}")

        # Save loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=features,
        )

        loadings_file = os.path.join(self.output_prefix, "pca_loadings.csv")
        loadings.to_csv(loadings_file)
        logger.info(f"Saved PCA loadings to {loadings_file}")

        # Save explained variance
        variance_df = pd.DataFrame(
            {
                "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            }
        )

        variance_file = os.path.join(self.output_prefix, "pca_variance.csv")
        variance_df.to_csv(variance_file, index=False)
        logger.info(f"Saved PCA variance to {variance_file}")

        self.results["pca"] = {
            "scores": pca_results,
            "loadings": loadings,
            "variance": variance_df,
        }

        logger.info(
            f"PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance"
        )
        logger.info(
            f"First 5 PCs explain {np.sum(pca.explained_variance_ratio_[:5])*100:.2f}% of variance"
        )

        return pca_results, loadings, variance_df

    def perform_random_forest_classification(self):
        """Train random forest classifier and extract feature importance."""
        logger.info("Training random forest classifier...")

        features = self.get_numeric_features()

        # Prepare data
        analysis_data = self.data[
            self.data["coding_class"].isin(["coding", "lncRNA"])
        ].copy()
        X = analysis_data[features].fillna(0)
        y = (analysis_data["coding_class"] == "lncRNA").astype(int)

        # Train random forest
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        # Cross-validation
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
        logger.info(
            f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )

        # Train on full data
        rf.fit(X, y)

        # Feature importance
        importance_df = pd.DataFrame(
            {"feature": features, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        output_file = os.path.join(self.output_prefix, "feature_importance.csv")
        importance_df.to_csv(output_file, index=False)
        logger.info(f"Saved feature importance to {output_file}")

        self.results["random_forest"] = {
            "cv_scores": cv_scores,
            "feature_importance": importance_df,
        }

        # Report top features
        logger.info("Top 10 most important features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        logger.info("Generating summary report...")

        report_file = os.path.join(self.output_prefix, "summary_report.txt")

        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TE FEATURE STATISTICAL ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total transcripts: {len(self.all_data)}\n")
            f.write(f"Coding transcripts: {len(self.coding_data)}\n")
            f.write(f"lncRNA transcripts: {len(self.lncrna_data)}\n\n")

            # TE presence
            coding_with_te = self.coding_data["hit_present"].sum()
            lncrna_with_te = self.lncrna_data["hit_present"].sum()

            f.write("TE PRESENCE\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Coding with TEs: {coding_with_te} ({coding_with_te/len(self.coding_data)*100:.1f}%)\n"
            )
            f.write(
                f"lncRNA with TEs: {lncrna_with_te} ({lncrna_with_te/len(self.lncrna_data)*100:.1f}%)\n\n"
            )

            # Significant features
            if "univariate_tests" in self.results:
                sig_features = self.results["univariate_tests"][
                    self.results["univariate_tests"]["mann_whitney_q"] < 0.05
                ]

                f.write("SIGNIFICANT FEATURES (FDR < 0.05)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Number of significant features: {len(sig_features)}\n\n")

                f.write("Top 10 most significant features:\n")
                for idx, row in sig_features.head(10).iterrows():
                    f.write(f"  {row['feature']}:\n")
                    f.write(f"    Coding mean: {row['coding_mean']:.3f}\n")
                    f.write(f"    lncRNA mean: {row['lncrna_mean']:.3f}\n")
                    f.write(f"    Fold change: {row['fold_change']:.3f}\n")
                    f.write(f"    p-value: {row['mann_whitney_p']:.2e}\n")
                    f.write(f"    q-value: {row['mann_whitney_q']:.2e}\n\n")

            # Random forest
            if "random_forest" in self.results:
                cv_scores = self.results["random_forest"]["cv_scores"]
                f.write("RANDOM FOREST CLASSIFICATION\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n\n"
                )

                f.write("Top 10 most important features for classification:\n")
                importance = self.results["random_forest"]["feature_importance"]
                for idx, row in importance.head(10).iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

        logger.info(f"Saved summary report to {report_file}")

    def run_all_analyses(self):
        """Run all statistical analyses."""
        self.load_data()
        self.perform_univariate_tests()
        self.perform_categorical_tests()
        self.perform_pca()
        self.perform_random_forest_classification()
        self.generate_summary_report()

        logger.info("All statistical analyses complete!")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of TE features")
    parser.add_argument("--features", required=True, help="TE features CSV file")
    parser.add_argument("--output-prefix", required=True, help="Output prefix")

    args = parser.parse_args()

    analyzer = TEStatisticalAnalyzer(
        features_file=args.features, output_prefix=args.output_prefix
    )

    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()
