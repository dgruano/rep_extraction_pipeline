#!/usr/bin/env python3
"""
TE Contingency Analyzer
=======================
Perform chi-square contingency tests for TE class and family enrichment
between protein-coding and lncRNA transcripts.

This script evaluates:
1. Overall TE presence/absence by coding class
2. TE class presence/absence by coding class
3. TE family presence/absence by coding class

Output includes contingency tables, chi-square statistics, p-values, and effect sizes.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TEContingencyAnalyzer:
    """Chi-square contingency analysis for TE enrichment between transcript groups."""

    # TE classes for organization
    TE_CLASSES = {
        "LINE": [
            "L2",
            "RTE-BovB",
            "L1",
            "CR1",
            "RTE-X",
            "Dong-R4",
            "I-Jockey",
            "L1-Tx1",
        ],
        "SINE": ["Alu", "MIR", "tRNA", "tRNA-RTE", "5S-Deu-L2", "tRNA-Deu"],
        "LTR": ["ERVL", "ERVL-MaLR", "ERV1", "ERVK", "DIRS", "Gypsy"],
        "DNA": [
            "TcMar-Tigger",
            "hAT-Charlie",
            "hAT-Tip100",
            "hAT-Blackjack",
            "TcMar-Mariner",
            "TcMar-Tc2",
            "hAT-Ac",
            "hAT",
            "PiggyBac",
            "MULE-MuDR",
            "hAT-Tag1",
            "TcMar-Tc1",
            "PIF-Harbinger",
            "Kolobok",
            "Merlin",
            "Crypton",
            "Crypton-A",
            "TcMar",
        ],
        "RC": ["Helitron"],
        "Retroposon": ["SVA"],
        "PLE": [],
    }

    def __init__(
        self, features_file: str, output_prefix: str, rm_out_file: Optional[str] = None
    ):
        """
        Initialize analyzer.

        Parameters
        ----------
        features_file : str
            Path to TE features CSV file (output from te_feature_extractor)
        output_prefix : str
            Output prefix for results files
        rm_out_file : str, optional
            Path to raw RepeatMasker .out file.  When provided, TE family
            presence is computed directly from hit_family fields instead of
            relying on pre-aggregated feature columns (which lack per-family
            columns for SINE/LINE families and have a bug for ERV families).
        """
        self.features_file = features_file
        self.output_prefix = output_prefix
        self.rm_out_file = rm_out_file

        # Ensure output directory exists
        Path(self.output_prefix).mkdir(parents=True, exist_ok=True)

        self.all_data = None
        self.data = None
        self.coding_data = None
        self.lncrna_data = None
        self.results = {}

    def load_data(self):
        """Load and prepare feature data."""
        logger.info(f"Loading feature data from {self.features_file}...")

        self.all_data = pd.read_csv(self.features_file)

        # Filter to coding vs lncRNA
        self.data = self.all_data[
            self.all_data["coding_class"].isin(["coding", "lncRNA"])
        ].copy()

        logger.info(
            f"Loaded {len(self.data)} transcripts with {len(self.data.columns)} features"
        )
        logger.info(f"Total transcripts: {len(self.all_data)}")

        self.coding_data = self.data[self.data["coding_class"] == "coding"]
        self.lncrna_data = self.data[self.data["coding_class"] == "lncRNA"]

        logger.info(f"Total pc transcripts: {len(self.coding_data)}")
        logger.info(f"Total lncRNA transcripts: {len(self.lncrna_data)}")

    def get_te_presence_columns(self) -> Dict[str, str]:
        """
        Identify TE presence columns in the feature matrix.

        Presence columns typically have pattern: 'te_count_*', 'line_count_*', etc.
        or boolean columns like 'has_te_*'

        Returns
        -------
        Dict[str, str]
            Dictionary mapping TE feature names to column names
        """
        presence_cols = {}

        # Look for various presence patterns
        for col in self.data.columns:
            col_lower = col.lower()

            # Skip non-TE columns
            if any(
                skip in col_lower
                for skip in [
                    "transcript_id",
                    "coding_class",
                    "chrom",
                    "length",
                    "start",
                    "end",
                ]
            ):
                continue

            # TE class columns (e.g., 'te_count_LINE', 'line_count', etc.)
            if any(
                pattern in col_lower
                for pattern in ["_count_", "_count", "_has_", "_present", "_num_"]
            ):
                # Extract feature name
                feature_name = (
                    col.replace("_count", "")
                    .replace("_has", "")
                    .replace("_present", "")
                    .replace("_num", "")
                )
                feature_name = feature_name.strip("_")
                presence_cols[feature_name] = col

        return presence_cols

    def create_binary_presence(self, col: str) -> pd.Series:
        """
        Convert a count/presence column to binary (present/absent).

        Parameters
        ----------
        col : str
            Column name

        Returns
        -------
        pd.Series
            Binary series (1 = present, 0 = absent)
        """
        return (self.data[col] > 0).astype(int)

    def perform_chi_square_test(
        self, feature_name: str, col: str, data: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Perform chi-square test for a single TE feature.

        Parameters
        ----------
        feature_name : str
            Name of the TE feature
        col : str
            Column name in data
        data : pd.DataFrame, optional
            DataFrame to use.  Defaults to self.data.  Must contain a
            'coding_class' column and the column referenced by `col`.

        Returns
        -------
        Dict or None
            Test results including chi2, p-value, effect size, or None if insufficient data
        """
        if data is None:
            data = self.data

        # Create binary presence
        presence = (data[col] > 0).astype(int)

        # Create contingency table: rows=coding_class, cols=presence(0,1)
        contingency = pd.crosstab(data["coding_class"], presence, margins=False)

        # Check for valid contingency table
        if contingency.shape != (2, 2):
            logger.warning(
                f"Invalid contingency table shape for {feature_name}: {contingency.shape}"
            )
            return None

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Calculate effect size (Cramér's V)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else 0

        # Extract counts (use `.get` so it works regardless whether 0/1 labels exist)
        coding_row = (
            contingency.loc["coding"]
            if "coding" in contingency.index
            else pd.Series({0: 0, 1: 0})
        )
        lncrna_row = (
            contingency.loc["lncRNA"]
            if "lncRNA" in contingency.index
            else pd.Series({0: 0, 1: 0})
        )
        coding_absent = coding_row.get(0, 0)
        coding_present = coding_row.get(1, 0)
        lncrna_absent = lncrna_row.get(0, 0)
        lncrna_present = lncrna_row.get(1, 0)

        coding_total = coding_absent + coding_present
        lncrna_total = lncrna_absent + lncrna_present

        # Calculate percentages
        pct_coding = (coding_present / coding_total * 100) if coding_total > 0 else 0
        pct_lncrna = (lncrna_present / lncrna_total * 100) if lncrna_total > 0 else 0

        return {
            "te_feature": feature_name,
            "pct_coding": pct_coding,
            "pct_lncrna": pct_lncrna,
            "coding_present": coding_present,
            "coding_total": coding_total,
            "lncrna_present": lncrna_present,
            "lncrna_total": lncrna_total,
            "chi2": chi2,
            "p_value": p_value,
            "cramers_v": cramers_v,
            "significant": p_value < 0.05,
        }

    def _parse_rm_family_presence(self) -> pd.DataFrame:
        """
        Parse the raw RepeatMasker .out file and compute per-transcript
        presence (0/1) for every unique hit_family.

        Returns
        -------
        pd.DataFrame
            Columns: transcript_id, coding_class, <family_*> …
            One row per transcript in self.data; families absent from a
            transcript are filled with 0.
        """
        logger.info(f"Parsing RepeatMasker families from {self.rm_out_file} …")

        col_names = [
            "sw_score",
            "perc_div",
            "perc_del",
            "perc_ins",
            "transcript_id",
            "start",
            "end",
            "query_left",
            "strand",
            "hit_name",
            "repeat_class_family",
            "repeat_begin",
            "repeat_end",
            "repeat_left",
            "id",
            "asterisk",
        ]
        rm = pd.read_csv(
            self.rm_out_file,
            sep=r"\s+",
            skiprows=3,
            names=col_names,
            engine="python",
            on_bad_lines="skip",
        )

        # Parse class/family split (e.g. "SINE/Alu" → class=SINE, family=Alu)
        cfsp = rm["repeat_class_family"].str.split("/", expand=True)
        rm["hit_class"] = cfsp[0]
        rm["hit_family"] = cfsp[1] if len(cfsp.columns) > 1 else cfsp[0]

        # Normalise: when family is missing, fall back to class
        rm["hit_family"] = rm["hit_family"].fillna(rm["hit_class"])

        # Keep only transcripts present in the filtered analysis set
        valid_ids = set(self.data["transcript_id"])
        rm = rm[rm["transcript_id"].isin(valid_ids)]

        # Drop rows with null family
        rm = rm[rm["hit_family"].notna() & (rm["hit_family"] != "")]

        logger.info(
            f"  {len(rm)} RM hits across "
            f"{rm['transcript_id'].nunique()} transcripts, "
            f"{rm['hit_family'].nunique()} unique families"
        )

        # Pivot: transcript × family → presence (0/1)
        presence = (
            rm.groupby(["transcript_id", "hit_family"])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)
            .astype(int)
        )
        presence.columns = [f"family_{c}" for c in presence.columns]
        presence = presence.reset_index()

        # Right-merge with full transcript list so absent transcripts get 0s
        analysis_ids = self.data[["transcript_id", "coding_class"]]
        presence = analysis_ids.merge(presence, on="transcript_id", how="left")
        family_cols = [c for c in presence.columns if c.startswith("family_")]
        presence[family_cols] = presence[family_cols].fillna(0).astype(int)

        return presence

    def perform_overall_te_presence_test(self):
        """
        Test for overall TE presence difference between coding and lncRNA.

        Returns
        -------
        pd.DataFrame
            Test results
        """
        logger.info("=" * 80)
        logger.info("Overall TE Presence: Coding vs lncRNA transcripts")
        logger.info("=" * 80)

        # Look for overall TE presence columns
        te_presence_cols = [
            col
            for col in self.data.columns
            if any(
                pattern in col.lower()
                for pattern in ["te_count", "te_present", "has_te"]
            )
            and "class" not in col.lower()
            and "family" not in col.lower()
        ]

        if not te_presence_cols:
            logger.warning("Could not find overall TE presence column")
            return None

        col = te_presence_cols[0]
        result = self.perform_chi_square_test("Overall_TE", col)

        if result:
            logger.info(
                f"\nPercentage of coding transcripts with TEs: {result['pct_coding']:.2f}%"
            )
            logger.info(
                f"Percentage of lncRNA transcripts with TEs: {result['pct_lncrna']:.2f}%"
            )
            logger.info(f"\nChi-square statistic: {result['chi2']:.4f}")
            logger.info(f"P-value: {result['p_value']:.4e}")
            logger.info(f"Significant difference: {result['significant']}")

        return pd.DataFrame([result]) if result else pd.DataFrame()

    def perform_te_class_tests(self):
        """
        Perform chi-square tests for each TE class.

        Returns
        -------
        pd.DataFrame
            Test results for all TE classes
        """
        logger.info("\n" + "=" * 80)
        logger.info(
            "Chi-square tests for TE class presence between coding and lncRNA transcripts"
        )
        logger.info("=" * 80)

        results = []

        # Get all unique TE classes in data
        class_cols = {}
        for col in self.data.columns:
            col_lower = col.lower()
            # Look for TE class columns (e.g., 'has_LINE', 'LINE_count', etc.)
            if any(
                pattern in col_lower
                for pattern in [
                    "_line",
                    "_sine",
                    "_ltr",
                    "_dna",
                    "_rc",
                    "_retroposon",
                    "_ple",
                    "_low_complexity",
                    "_simple_repeat",
                    "_satellite",
                ]
            ):
                # Extract class name
                for pattern in [
                    "_line",
                    "_sine",
                    "_ltr",
                    "_dna",
                    "_rc",
                    "_retroposon",
                    "_ple",
                    "_low_complexity",
                    "_simple_repeat",
                    "_satellite",
                ]:
                    if pattern in col_lower:
                        class_name = (
                            col.split(pattern)[0]
                            .replace("te_", "")
                            .replace("has_", "")
                            .strip("_")
                        )
                        if pattern == "_low_complexity":
                            class_name = "Low_complexity"
                        elif pattern == "_simple_repeat":
                            class_name = "Simple_repeat"
                        elif pattern == "_satellite":
                            class_name = "Satellite"
                        else:
                            class_name = pattern.strip("_").upper()

                        class_cols[class_name] = col
                        break

        # Perform tests
        for class_name in sorted(class_cols.keys()):
            col = class_cols[class_name]
            result = self.perform_chi_square_test(class_name, col)
            if result:
                results.append(result)
                logger.info(
                    f"{class_name}: p={result['p_value']:.4e}, significant={result['significant']}"
                )

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Sort by p-value
            results_df = results_df.sort_values("p_value")

            # Save to file
            output_file = os.path.join(self.output_prefix, "te_class_chi_square.csv")
            results_df.to_csv(output_file, index=False)
            logger.info(f"\nSaved TE class results to {output_file}")

            # Print summary
            sig_count = (results_df["p_value"] < 0.05).sum()
            logger.info(f"\nSignificant TE classes: {sig_count} / {len(results_df)}")

        self.results["te_classes"] = results_df
        return results_df

    def perform_te_family_tests(self):
        """
        Perform chi-square tests for each TE family.

        When ``self.rm_out_file`` is set, family presence is derived directly
        from the raw RepeatMasker .out file (using the ``hit_family`` field).
        This is preferred because the pre-aggregated feature columns do not
        include per-family columns for SINE/LINE families and contain a bug
        for ERV families (hit_class is always the top-level class, not the
        ERV sub-type).

        Without ``rm_out_file``, the method falls back to detecting
        ``_count``/``_has_`` columns in the features CSV (legacy behaviour).

        Returns
        -------
        pd.DataFrame
            Test results for all TE families
        """
        logger.info("\n" + "=" * 80)
        logger.info(
            "Chi-square tests for TE family presence between coding and lncRNA transcripts"
        )
        logger.info("=" * 80)

        results = []

        # ---------------------------------------------------------------
        # PATH A: derive family presence from raw RM output (preferred)
        # ---------------------------------------------------------------
        if self.rm_out_file:
            family_presence = self._parse_rm_family_presence()
            family_cols = [
                c for c in family_presence.columns if c.startswith("family_")
            ]

            for col in family_cols:
                family_name = col.replace("family_", "")
                result = self.perform_chi_square_test(
                    family_name, col, data=family_presence
                )
                if result:
                    results.append(result)
                    logger.info(
                        f"{family_name}: p={result['p_value']:.4e}, significant={result['significant']}"
                    )

        # ---------------------------------------------------------------
        # PATH B: legacy column-detection from features CSV
        # ---------------------------------------------------------------
        else:
            logger.warning(
                "No --rm-out file provided; falling back to feature-column detection. "
                "ERV sub-families and most SINE/LINE families will be missing."
            )
            family_col_map = {}
            for col in self.data.columns:
                col_lower = col.lower()
                if any(
                    skip in col_lower
                    for skip in [
                        "transcript_id",
                        "coding_class",
                        "chrom",
                        "length",
                        "start",
                        "end",
                        "_class",
                        "_count_",
                        "_gap",
                        "_num_fragments",
                    ]
                ):
                    continue
                if any(
                    pattern in col_lower
                    for pattern in ["_count", "_has_", "_present", "_num_"]
                ):
                    family_name = (
                        col.replace("_count", "")
                        .replace("_has", "")
                        .replace("_present", "")
                        .replace("_num", "")
                    )
                    family_name = family_name.strip("_").replace("te_", "")
                    skip_names = [
                        "line",
                        "sine",
                        "ltr",
                        "dna",
                        "rc",
                        "retroposon",
                        "ple",
                        "low_complexity",
                        "simple_repeat",
                        "satellite",
                        "overall",
                        "te",
                    ]
                    if family_name.lower() not in skip_names:
                        family_col_map[family_name] = col

            for family_name in sorted(family_col_map.keys()):
                col = family_col_map[family_name]
                result = self.perform_chi_square_test(family_name, col)
                if result:
                    results.append(result)
                    logger.info(
                        f"{family_name}: p={result['p_value']:.4e}, significant={result['significant']}"
                    )

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Sort by chi2 descending (most significant first)
            results_df = results_df.sort_values("chi2", ascending=False)

            # Save to file
            output_file = os.path.join(self.output_prefix, "te_family_chi_square.csv")
            results_df.to_csv(output_file, index=False)
            logger.info(f"\nSaved TE family results to {output_file}")

            # Print summary
            sig_count = (results_df["p_value"] < 0.05).sum()
            logger.info(f"\nSignificant TE families: {sig_count} / {len(results_df)}")

        self.results["te_families"] = results_df
        return results_df

    def generate_report(self):
        """Generate comprehensive text report, print to stdout, and save to file."""
        sep = "=" * 80
        lines = []

        # Header counts (outside any section)
        lines.append(f"Total pc transcripts: {len(self.coding_data)}")
        lines.append(f"Total lncRNA transcripts: {len(self.lncrna_data)}")

        # Overall TE presence section
        if "overall_te" in self.results:
            overall_df = self.results["overall_te"]
            lines.append(sep)
            lines.append("Overall TE Presence: Coding vs lncRNA transcripts")
            lines.append(sep)
            lines.append("")
            if not overall_df.empty:
                row = overall_df.iloc[0]
                lines.append(
                    f"Percentage of coding transcripts with TEs: {row['pct_coding']:.2f}%"
                )
                lines.append(
                    f"Percentage of lncRNA transcripts with TEs: {row['pct_lncrna']:.2f}%"
                )
                lines.append("")
                lines.append(f"Chi-square statistic: {row['chi2']:.4f}")
                lines.append(f"P-value: {row['p_value']:.4e}")
                lines.append(f"Significant difference: {row['significant']}")
            lines.append("")

        # TE class section
        if "te_classes" in self.results and not self.results["te_classes"].empty:
            lines.append(sep)
            lines.append(
                "Chi-square tests for TE class presence between coding and lncRNA transcripts"
            )
            lines.append(sep)
            class_df = self.results["te_classes"][
                [
                    "te_feature",
                    "pct_coding",
                    "pct_lncrna",
                    "chi2",
                    "p_value",
                    "significant",
                ]
            ]
            lines.append(class_df.to_string(index=False))
            lines.append("")

        # TE family section
        if "te_families" in self.results and not self.results["te_families"].empty:
            lines.append(sep)
            lines.append(
                "Chi-square tests for TE family presence between coding and lncRNA transcripts"
            )
            lines.append(sep)
            family_df = self.results["te_families"][
                [
                    "te_feature",
                    "pct_coding",
                    "pct_lncrna",
                    "chi2",
                    "p_value",
                    "significant",
                ]
            ]
            lines.append(family_df.to_string(index=False))
            lines.append("")

        # Consolidated SUMMARY section
        lines.append(sep)
        lines.append("SUMMARY")
        lines.append(sep)
        lines.append("")
        if "te_classes" in self.results and not self.results["te_classes"].empty:
            sig_count = (self.results["te_classes"]["p_value"] < 0.05).sum()
            lines.append(
                f"TE Classes - Significant differences: {sig_count} / {len(self.results['te_classes'])}"
            )
        if "te_families" in self.results and not self.results["te_families"].empty:
            sig_count = (self.results["te_families"]["p_value"] < 0.05).sum()
            lines.append(
                f"TE Families - Significant differences: {sig_count} / {len(self.results['te_families'])}"
            )

        report_text = "\n".join(lines)

        # Print to stdout
        print(report_text)

        # Save to file
        report_file = os.path.join(
            self.output_prefix, "contingency_analysis_report.txt"
        )
        with open(report_file, "w") as f:
            f.write(report_text + "\n")

        logger.info(f"\nSaved report to {report_file}")

    def run_all_analyses(self):
        """Run all contingency analyses."""
        self.load_data()

        # Overall TE presence
        overall_results = self.perform_overall_te_presence_test()
        self.results["overall_te"] = overall_results

        # TE classes
        self.perform_te_class_tests()

        # TE families
        self.perform_te_family_tests()

        # Generate report
        self.generate_report()

        logger.info("\n" + "=" * 80)
        logger.info("All contingency analyses complete!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Chi-square contingency analysis of TE enrichment between coding and lncRNA transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --features features.csv --output-prefix results/

  # With output directory creation
  %(prog)s --features /path/to/features.csv --output-prefix /path/to/output/
        """,
    )
    parser.add_argument(
        "--features",
        required=True,
        help="TE features CSV file (from te_feature_extractor)",
    )
    parser.add_argument(
        "--output-prefix", required=True, help="Output prefix (directory) for results"
    )
    parser.add_argument(
        "--rm-out",
        default=None,
        help=(
            "Raw RepeatMasker .out file.  When provided, TE family "
            "presence is computed directly from hit_family fields, "
            "enabling per-family tests for SINE (Alu, MIR, tRNA…), "
            "LINE (L1, L2, CR1…), LTR (ERV1, ERVL-MaLR, Gypsy…), "
            "DNA transposons, and other families. "
            "Strongly recommended; without this flag ERV and most SINE/LINE families will be absent."
        ),
    )

    args = parser.parse_args()

    analyzer = TEContingencyAnalyzer(
        features_file=args.features,
        output_prefix=args.output_prefix,
        rm_out_file=args.rm_out,
    )

    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()
