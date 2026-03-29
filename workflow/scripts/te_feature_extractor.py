#!/usr/bin/env python3
"""
TE Feature Extractor - Refactored Modular Version
==================================================
Extract TE features with biotype tracking from full transcript database.

REFACTORED: Modular architecture separating:
- Data loading (RepeatMasker format handlers)
- Overlap detection and filtering
- Feature extraction by repeat type (TE, Low_complexity, Tandem repeats)
- Metadata integration
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# Configure logging to work with tqdm
class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid interfering with progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


# Determine if output should be interactive
USE_TQDM = sys.stderr.isatty()

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = TqdmLoggingHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def log_progress(current, total, desc="Processing"):
    """Log progress information for non-interactive sessions."""
    if not USE_TQDM:
        if current % max(1, total // 10) == 0 or current == total:
            pct = (current / total) * 100 if total > 0 else 0
            logger.info(f"{desc}: {current}/{total} ({pct:.1f}%)")


# =============================================================================
# DATA LOADING - RepeatMasker .out Format Handler
# =============================================================================


class OutFileLoader:
    """Load RepeatMasker .out format."""

    def load(self, filepath: str) -> pd.DataFrame:
        """Load RepeatMasker .out file."""
        logger.info(f"Loading .out format from {filepath}")

        # Read file and find data start
        with open(filepath, "r") as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(" ") and "SW" not in line:
                data_start = i + 1
                break
            if "ID" in line:
                data_start = i + 2
                break

        # Column names for RepeatMasker .out format
        col_names = [
            "sw_score",
            "perc_div",
            "perc_del",
            "perc_ins",
            "query_sequence",
            "query_begin",
            "query_end",
            "query_left",
            "strand",
            "repeat_name",
            "repeat_class_family",
            "repeat_begin",
            "repeat_end",
            "repeat_left",
            "id",
            "asterisk",
        ]

        rm_hits = pd.read_csv(
            filepath, sep=r"\s+", skiprows=data_start, names=col_names, engine="python"
        )

        # Clean up columns
        rm_hits["query_sequence"] = rm_hits["query_sequence"].astype(str)

        # Numeric columns
        int_cols = ["query_begin", "query_end", "repeat_begin", "repeat_end"]
        for c in int_cols:
            rm_hits[c] = pd.to_numeric(rm_hits[c], errors="coerce").astype("Int64")

        # Parse parenthesized integers
        parse_int_cols = ["query_left", "repeat_left"]
        for c in parse_int_cols:
            rm_hits[c] = pd.to_numeric(
                rm_hits[c].astype(str).str.strip("()"), errors="coerce"
            ).astype("Int64")

        # Parse strand
        rm_hits["strand"] = rm_hits["strand"].replace({"C": "-", "+": "+"})

        # Split repeat_class_family
        class_family_split = rm_hits["repeat_class_family"].str.split("/", expand=True)
        rm_hits["hit_class"] = (
            class_family_split[0] if len(class_family_split.columns) > 0 else None
        )
        rm_hits["hit_family"] = (
            class_family_split[1] if len(class_family_split.columns) > 1 else None
        )

        # Rename to standard schema
        rm_hits = rm_hits.rename(
            columns={
                "query_sequence": "transcript_id",
                "query_begin": "start",
                "query_end": "end",
                "repeat_name": "hit_name",
                "perc_div": "divergence",
            }
        )

        # Calculate hit length
        rm_hits["hit_length"] = rm_hits["end"] - rm_hits["start"] + 1

        # Normalize transcript ID (remove version if present)
        # rm_hits['transcript_id'] = rm_hits['transcript_id'].str.split('.').str[0]

        return rm_hits


def load_repeatmasker_out(filepath: str) -> pd.DataFrame:
    """
    Load RepeatMasker .out format file.

    Parameters
    ----------
    filepath : str
        Path to RepeatMasker .out file

    Returns
    -------
    pd.DataFrame
        RepeatMasker hits with standardized columns
    """
    logger.info(f"Loading RepeatMasker .out format from {filepath}")
    loader = OutFileLoader()
    return loader.load(filepath)


# =============================================================================
# OVERLAP DETECTION AND FILTERING
# =============================================================================


class OverlapFilter:
    """Detect and resolve overlapping RepeatMasker hits."""

    @staticmethod
    def find_overlapping_hits(
        rm_hits: pd.DataFrame,
        group_col: str = "transcript_id",
        start_col: str = "start",
        end_col: str = "end",
        extended_output: bool = True,
    ) -> pd.DataFrame:
        """
        Find all pairwise overlapping intervals within groups using sweep-line algorithm.

        Parameters
        ----------
        rm_hits : pd.DataFrame
            RepeatMasker hits DataFrame
        group_col : str
            Column to group by (default: 'transcript_id')
        start_col : str
            Start coordinate column (default: 'start')
        end_col : str
            End coordinate column (default: 'end')
        extended_output : bool
            If True, include all columns from both overlapping hits

        Returns
        -------
        pd.DataFrame
            DataFrame with overlapping pairs, containing:
            - overlap_start, overlap_end, overlap_length
            - index_1, index_2 (original DataFrame indices)
            - All columns from rm_hits with _1 and _2 suffixes (if extended_output=True)
        """
        overlaps = []

        for group_name, group_df in rm_hits.groupby(
            group_col, observed=True, sort=False
        ):
            # Sort by start for sweep-line algorithm
            group_df = group_df.sort_values(start_col).reset_index(drop=False)
            orig_indices = group_df["index"].values

            starts = group_df[start_col].values
            ends = group_df[end_col].values
            n = len(starts)

            other_cols = [
                col for col in group_df.columns if col not in [group_col, "index"]
            ]

            # Sweep-line algorithm with early termination
            for i in range(n - 1):
                start1, end1 = starts[i], ends[i]

                for j in range(i + 1, n):
                    start2 = starts[j]

                    # Early termination: no more overlaps possible
                    if start2 > end1:
                        break

                    end2 = ends[j]
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)

                    if overlap_start <= overlap_end:
                        # Build overlap record with columns from both hits
                        overlap_record = {
                            group_col: group_name,
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_length": overlap_end - overlap_start,
                            "index_1": orig_indices[i],
                            "index_2": orig_indices[j],
                        }

                        if extended_output:
                            for col in other_cols:
                                overlap_record[f"{col}_1"] = group_df.at[i, col]
                                overlap_record[f"{col}_2"] = group_df.at[j, col]

                        overlaps.append(overlap_record)

        return pd.DataFrame(overlaps)

    @staticmethod
    def filter_overlapping_hits(
        rm_hits: pd.DataFrame, overlaps: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Resolve overlapping hits using hierarchical filtering rules.

        Rules (applied in order):
        1. Remove hits with asterisk (RepeatMasker flag for better match exists)
        2. Keep hit with higher SW score
        3. Keep hit with lower overlap percentage (more unique)
        4. Default: remove index_2

        Parameters
        ----------
        rm_hits : pd.DataFrame
            RepeatMasker hits DataFrame
        overlaps : pd.DataFrame
            Overlapping pairs from find_overlapping_hits()

        Returns
        -------
        pd.DataFrame
            Filtered rm_hits with overlapping duplicates removed
        """
        if overlaps.empty:
            logger.info("No overlapping TE hits found.")
            return rm_hits

        logger.info(f"Found {len(overlaps)} overlapping TE hit pairs.")

        indices_to_remove = set()

        # Rule 0: Remove hits with asterisk
        mask_1_has_asterisk = overlaps["asterisk_1"].notna()
        mask_2_has_asterisk = overlaps["asterisk_2"].notna()
        indices_to_remove.update(overlaps.loc[mask_1_has_asterisk, "index_1"].tolist())
        indices_to_remove.update(overlaps.loc[mask_2_has_asterisk, "index_2"].tolist())

        # Rule 1: Keep higher SW score (only when neither has asterisk)
        mask_no_asterisk = ~(mask_1_has_asterisk | mask_2_has_asterisk)
        mask_score_1_lower = mask_no_asterisk & (
            overlaps["sw_score_1"] < overlaps["sw_score_2"]
        )
        mask_score_2_lower = mask_no_asterisk & (
            overlaps["sw_score_2"] < overlaps["sw_score_1"]
        )
        indices_to_remove.update(overlaps.loc[mask_score_1_lower, "index_1"].tolist())
        indices_to_remove.update(overlaps.loc[mask_score_2_lower, "index_2"].tolist())

        # Rule 2: Keep hit with lower overlap percentage (when scores equal)
        overlaps["perc_1"] = (
            overlaps["overlap_length"] / overlaps["hit_length_1"]
        ) * 100
        overlaps["perc_2"] = (
            overlaps["overlap_length"] / overlaps["hit_length_2"]
        ) * 100
        mask_equal_score = mask_no_asterisk & (
            overlaps["sw_score_1"] == overlaps["sw_score_2"]
        )
        mask_perc_1_higher = mask_equal_score & (
            overlaps["perc_1"] > overlaps["perc_2"]
        )
        mask_perc_2_higher = mask_equal_score & (
            overlaps["perc_2"] > overlaps["perc_1"]
        )
        indices_to_remove.update(overlaps.loc[mask_perc_1_higher, "index_1"].tolist())
        indices_to_remove.update(overlaps.loc[mask_perc_2_higher, "index_2"].tolist())

        # Rule 3: Default removal (equal everything)
        mask_remaining = mask_equal_score & (overlaps["perc_1"] == overlaps["perc_2"])
        indices_to_remove.update(overlaps.loc[mask_remaining, "index_2"].tolist())

        # Log statistics
        logger.info(
            f"Pairs with asterisk: {(mask_1_has_asterisk | mask_2_has_asterisk).sum()}"
        )
        logger.info(
            f"Pairs resolved by score: {(mask_score_1_lower | mask_score_2_lower).sum()}"
        )
        logger.info(
            f"Pairs resolved by overlap %: {(mask_perc_1_higher | mask_perc_2_higher).sum()}"
        )
        logger.info(f"Total unique indices to remove: {len(indices_to_remove)}")

        # Remove identified indices
        filtered = rm_hits.drop(index=list(indices_to_remove)).reset_index(drop=True)
        logger.info(f"Retained {len(filtered)}/{len(rm_hits)} hits after filtering")

        return filtered


# =============================================================================
# BASE PROCESSOR - Shared Utilities
# =============================================================================


class RepeatMaskerProcessor:
    """Base class with shared utilities for processing RepeatMasker annotations."""

    def __init__(self, rm_hits: pd.DataFrame, transcripts: pd.DataFrame):
        """
        Initialize processor.

        Parameters
        ----------
        rm_hits : pd.DataFrame
            RepeatMasker hits (after overlap filtering)
        transcripts : pd.DataFrame
            Transcript coordinates with 'transcript_id' and 'length' columns
        """
        self.rm_hits = rm_hits.copy()
        self.transcripts = transcripts

    @staticmethod
    def calculate_hit_lengths(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate length-related features for RepeatMasker hits.

        Adds columns:
        - hit_length: Length on transcript (target span)
        - hit_repeat_length: Length on reference repeat (query span)
        - hit_ref_length: Total reference repeat length
        - hit_reference_coverage: Fraction of reference covered

        Parameters
        ----------
        df : pd.DataFrame
            RepeatMasker hits with columns:
            start, end, repeat_begin, repeat_end, repeat_left, strand

        Returns
        -------
        pd.DataFrame
            Input DataFrame with added length columns
        """
        # Check presence of required columns
        required = [
            "start",
            "end",
            "repeat_begin",
            "repeat_end",
            "repeat_left",
            "strand",
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.copy()
        df["hit_length"] = df["end"] - df["start"] + 1

        # Strand-specific calculations
        df.loc[df["strand"] == "+", "hit_repeat_length"] = (
            df["repeat_end"] - df["repeat_begin"] + 1
        )  # e.g., 10, 200, (50) -> 200 - 10 + 1 = 191
        df.loc[df["strand"] == "-", "hit_repeat_length"] = (
            df["repeat_end"] - df["repeat_left"] + 1
        )  # e.g., (50), 200, 10 -> 200 - 10 + 1 =  -149

        df.loc[df["strand"] == "+", "hit_ref_length"] = (
            df["repeat_end"] + df["repeat_left"]
        )  # e.g., 10, 200, (50) -> 200 + 50 = 250
        df.loc[df["strand"] == "-", "hit_ref_length"] = (
            df["repeat_end"] + df["repeat_begin"]
        )  # e.g., (50), 200, 10 -> 200 + 50 = 250

        df["hit_reference_coverage"] = df["hit_repeat_length"] / df["hit_ref_length"]

        return df

    @staticmethod
    def calculate_gaps_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized gap calculation (10-12x faster than groupby.apply).

        Calculates gaps between consecutive hits within each transcript:
        - gap_before: distance from hit start to previous hit end
        - gap_after: distance from hit end to next hit start

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: transcript_id, start, end, query_left
            MUST be sorted by (transcript_id, start)

        Returns
        -------
        pd.DataFrame
            Input DataFrame with gap_before and gap_after columns added
        """
        df = df.copy()

        # Convert query_left to numeric
        if df["query_left"].dtype == object:
            df["query_left_num"] = (
                pd.to_numeric(
                    df["query_left"].astype(str).str.strip("()"), errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )
        else:
            df["query_left_num"] = df["query_left"].astype(int)

        # Identify first and last elements per transcript
        df["first_element"] = df["transcript_id"].ne(df["transcript_id"].shift(1))
        df["last_element"] = df["first_element"].shift(-1).fillna(True)

        # Calculate gaps using vectorized shift
        df["gap_before"] = df["start"] - df["end"].shift(1) - 1
        df.loc[df["first_element"], "gap_before"] = (
            df["start"] - 1
        )  # Gap before first hit

        df["gap_after"] = df["gap_before"].shift(-1)
        df.loc[df["last_element"], "gap_after"] = df["query_left"]  # Gap after last hit

        return df

    @staticmethod
    def calculate_gap_stats(group: pd.DataFrame) -> pd.Series:
        """
        Calculate gap statistics for groupby.apply().

        Parameters
        ----------
        group : pd.DataFrame
            Group of hits with gap_before and gap_after columns

        Returns
        -------
        pd.Series
            Gap statistics: mean, median, max, min
        """
        gaps = pd.concat([group["gap_before"], group["gap_after"]])

        return pd.Series(
            {
                "gaps_mean": gaps.mean(),
                "gaps_median": gaps.median(),
                "gaps_max": gaps.max(),
                "gaps_min": gaps.min(),
            }
        )

    @staticmethod
    def calculate_gap_stats_vectorized(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """
        Calculate gap statistics in a vectorized manner.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with gap_before and gap_after columns
        prefix : str
            Prefix for output column names

        Returns
        -------
        pd.DataFrame
            DataFrame with transcript_id and gap statistics columns
        """
        gaps_before = df[["transcript_id", "gap_before"]].rename(
            columns={"gap_before": "gap_size"}
        )
        is_last_hit = ~df["transcript_id"].duplicated(keep="last")
        gaps_after_last = df.loc[is_last_hit, ["transcript_id", "gap_after"]].rename(
            columns={"gap_after": "gap_size"}
        )
        combined = pd.concat([gaps_before, gaps_after_last], ignore_index=True).dropna(
            subset=["gap_size"]
        )

        stats = (
            combined.groupby("transcript_id", observed=True)["gap_size"]
            .agg(["mean", "median", "max", "min"])
            .reset_index()
        )
        stats.columns = [
            "transcript_id",
            f"{prefix}_gaps_mean",
            f"{prefix}_gaps_median",
            f"{prefix}_gaps_max",
            f"{prefix}_gaps_min",
        ]

        return stats


# =============================================================================
# SPECIALIZED PROCESSORS - Feature Extraction by Repeat Type
# =============================================================================


class TEProcessor(RepeatMaskerProcessor):
    """
    Process Transposable Elements (LINE, SINE, LTR, DNA, RC, Retroposon).

    TE-specific features:
    - Divergence-based age classification
    - Fragmentation patterns
    - Family/subfamily diversity
    - Class-specific distributions
    """

    # TE classes and families
    TE_CLASSES = {
        "DNA": [
            "Crypton",
            "Crypton-A",
            "hAT",
            "hAT-Ac",
            "hAT-Blackjack",
            "hAT-Charlie",
            "hAT-Tag1",
            "hAT-Tip100",
            "Kolobok",
            "Merlin",
            "MULE-MuDR",
            "PIF-Harbinger",
            "PiggyBac",
            "TcMar",
            "TcMar-Mariner",
            "TcMar-Tc1",
            "TcMar-Tc2",
            "TcMar-Tigger",
        ],
        "ERVK": [],
        "ERVL": [],
        "ERVL-MaLR": [],
        "ERV1": [],
        "LINE": [
            "CR1",
            "Dong-R4",
            "I-Jockey",
            "L1",
            "L1-Tx1",
            "L2",
            "RTE-BovB",
            "RTE-X",
        ],
        "LTR": ["DIRS", "ERVK", "ERVL", "ERVL-MaLR", "ERV1", "Gypsy"],
        "PLE": [],
        "RC": ["Helitron"],
        "Retroposon": ["SVA"],
        "SINE": ["5S-Deu-L2", "Alu", "MIR", "tRNA", "tRNA-Deu", "tRNA-RTE"],
        "srpRNA": [],  # TODO: Determine correct class. Dfam does not describe it, but Alus are derived from 7SLRNA (srpRNA)
    }

    # TODO: Check custom young subfamilies
    YOUNG_SUBFAMILIES = {
        "L1HS",
        "L1PA1",
        "L1PA2",
        "AluY",
        "AluYa",
        "AluYb",
        "AluYc",
        "SVA_F",
        "HERVK",
    }

    def extract_features(self) -> pd.DataFrame:
        """
        Extract TE-specific features at transcript level.

        Returns
        -------
        pd.DataFrame
            Transcript-level TE features with columns:
            - Divergence stats (min, max, mean)
            - Length stats (count, sum, mean, etc.)
            - Fragmentation metrics
            - Class-specific counts (per TE class)
            - Family diversity
            - Age classification (young/ancient)
        """
        logger.info("Extracting TE-specific features...")

        # Filter to TE classes only
        te_hits = self.rm_hits[
            self.rm_hits["hit_class"].isin(self.TE_CLASSES.keys())
        ].copy()

        if te_hits.empty:
            logger.warning("No TE hits found!")
            return self._empty_features()

        logger.info(
            f"Processing {len(te_hits)} TE hits in \
                    {te_hits['transcript_id'].nunique()} transcripts."
        )

        # Calculate basic metrics
        te_hits = self.calculate_hit_lengths(te_hits)
        te_hits = self.calculate_gaps_vectorized(te_hits)

        # Aggregate by element (fragments → elements)
        by_element = self._aggregate_by_element(te_hits)

        # Aggregate by transcript (elements → transcripts)
        by_transcript = self._aggregate_by_transcript(by_element)

        # Calculate gap statistics from hits directly
        gap_stats = self.calculate_gap_stats_vectorized(te_hits, "te")
        by_transcript = by_transcript.merge(gap_stats, on="transcript_id", how="left")

        return by_transcript

    def _aggregate_by_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate TE fragments into elements.

        Parameters
        ----------
        df : pd.DataFrame
            TE hits with calculated lengths and gaps

        Returns
        -------
        pd.DataFrame
            Element-level features
        """
        grouped = df.groupby("id")

        agg_dict = {
            "transcript_id": "first",
            "hit_length": "sum",
            "hit_reference_coverage": "sum",
            "sw_score": "first",
            "divergence": "first",  # Meaningful for TEs
            "perc_del": "first",
            "perc_ins": "first",
            "strand": "first",
            "hit_name": "first",
            "hit_family": "first",
            "hit_class": "first",
        }

        result = grouped.agg(agg_dict).reset_index()
        result["num_fragments"] = grouped.size().values
        result["fragmented"] = result["num_fragments"] > 1

        # Add class boolean columns
        for te_class in self.TE_CLASSES.keys():
            result[f"is_{te_class.lower()}"] = result["hit_class"] == te_class

        # Age classification
        result["is_young"] = result["hit_name"].isin(self.YOUNG_SUBFAMILIES)
        result["is_ancient"] = result["divergence"].astype(float) > 20

        return result

    def _aggregate_by_transcript(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate elements into transcript-level features.

        Parameters
        ----------
        df : pd.DataFrame
            Element-level features

        Returns
        -------
        pd.DataFrame
            Transcript-level TE features
        """
        grouped = df.groupby("transcript_id", observed=True)

        # Build aggregation dictionary
        agg_dict = {
            # Divergence stats (age)
            "sw_score": ["min", "max", "mean"],
            "divergence": ["min", "max", "mean"],
            "perc_del": ["min", "max", "mean"],
            "perc_ins": ["min", "max", "mean"],
            # Length stats
            "hit_length": ["count", "sum", "min", "mean", "max"],
            "hit_reference_coverage": ["min", "mean", "max"],
            "num_fragments": ["sum", "mean", "max"],
            "fragmented": "sum",
            # Diversity
            "hit_name": "nunique",
            "hit_class": "nunique",
            "hit_family": "nunique",
        }

        # Add class-specific aggregations
        for te_class in self.TE_CLASSES.keys():
            class_col = f"is_{te_class.lower()}"
            agg_dict[class_col] = ["max", "sum"]

        # Age aggregations
        agg_dict["is_young"] = "sum"
        agg_dict["is_ancient"] = "sum"

        result = grouped.agg(agg_dict).reset_index()

        # Flatten multi-index column names
        new_cols = ["transcript_id"]
        for col in result.columns[1:]:
            if isinstance(col, tuple):
                if col[1] == "count":
                    new_cols.append(f"te_{col[1]}")
                elif col[1] == "sum" and col[0].startswith("is_"):
                    # is_line sum → line_count
                    new_cols.append(f'te_{col[0].replace("is_", "")}_count')
                elif col[1] == "max" and col[0].startswith("is_"):
                    # is_line max → has_line
                    new_cols.append(f'te_has_{col[0].replace("is_", "")}')
                elif col[0] == "hit_name" and col[1] == "nunique":
                    new_cols.append("te_unique_subfamilies")
                elif col[0] == "hit_class" and col[1] == "nunique":
                    new_cols.append("te_unique_classes")
                elif col[0] == "hit_family" and col[1] == "nunique":
                    new_cols.append("te_unique_families")
                else:
                    new_cols.append(f"te_{col[1]}_{col[0]}")
            else:
                new_cols.append(col)

        result.columns = new_cols

        # Calculate derived relative metrics
        result["te_fragmented_ratio"] = result["te_sum_fragmented"] / result["te_count"]

        return result

    def _empty_features(self) -> pd.DataFrame:
        """Return empty DataFrame with expected columns."""
        return pd.DataFrame(columns=["transcript_id"])


class LowComplexityTandemProcessor(RepeatMaskerProcessor):
    """
    Process Low_complexity and Tandem Repeats (Simple_repeat, Satellite).

    LC/TR-specific features:
    - Coverage metrics (divergence not meaningful)
    - Class distribution
    - NO family diversity (most lack families)
    """

    # LC/TR classes (note: no families for most)
    LCTR_CLASSES = {
        "Low_complexity": [],
        "Simple_repeat": [],
        "Satellite": ["centr", "subtelo", "acro"],  # Some satellites have subfamilies
    }

    def extract_features(self) -> pd.DataFrame:
        """
        Extract LC/TR features at transcript level.

        Returns
        -------
        pd.DataFrame
            Transcript-level LC/TR features with columns:
            - Count and total length
            - Class-specific counts
            - NO divergence stats (not meaningful)
        """
        logger.info("Extracting Low_complexity/Tandem Repeat features...")

        # Filter to LC/TR classes
        lctr_hits = self.rm_hits[
            self.rm_hits["hit_class"].isin(self.LCTR_CLASSES.keys())
        ].copy()

        if lctr_hits.empty:
            logger.warning("No LC/TR hits found!")
            return self._empty_features()

        logger.info(
            f"Processing {len(lctr_hits)} LC/TR hits in \
                    {lctr_hits['transcript_id'].nunique()} transcripts."
        )

        # Calculate basic metrics (but not divergence)
        lctr_hits = self.calculate_hit_lengths(lctr_hits)
        lctr_hits = self.calculate_gaps_vectorized(lctr_hits)

        # Aggregate by element
        by_element = self._aggregate_by_element(lctr_hits)

        # Aggregate by transcript
        by_transcript = self._aggregate_by_transcript(by_element)

        # Calculate gap statistics
        gap_stats = self.calculate_gap_stats_vectorized(lctr_hits, "lctr")
        by_transcript = by_transcript.merge(gap_stats, on="transcript_id", how="left")

        return by_transcript

    def _aggregate_by_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate LC/TR fragments into elements."""
        grouped = df.groupby("id")

        agg_dict = {
            "transcript_id": "first",
            "hit_length": "sum",
            "hit_class": "first",
            "hit_name": "first",
        }

        result = grouped.agg(agg_dict).reset_index()
        result["num_fragments"] = grouped.size().values

        # Add class boolean columns
        for lctr_class in self.LCTR_CLASSES.keys():
            result[f"is_{lctr_class.lower()}"] = result["hit_class"] == lctr_class

        return result

    def _aggregate_by_transcript(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate elements into transcript-level LC/TR features."""
        grouped = df.groupby("transcript_id", observed=True)

        agg_dict = {
            "hit_length": ["count", "sum", "mean"],
            "num_fragments": ["sum", "mean"],
        }

        # Class-specific aggregations
        for lctr_class in self.LCTR_CLASSES.keys():
            class_col = f"is_{lctr_class.lower()}"
            agg_dict[class_col] = ["max", "sum"]

        result = grouped.agg(agg_dict).reset_index()

        # Flatten column names
        new_cols = ["transcript_id"]
        for col in result.columns[1:]:
            if isinstance(col, tuple):
                if col[1] == "count":
                    new_cols.append("lctr_count")
                elif col[1] == "sum" and col[0] == "hit_length":
                    new_cols.append("lctr_total_length")
                elif col[1] == "mean" and col[0] == "hit_length":
                    new_cols.append("lctr_mean_length")
                elif col[1] == "sum" and col[0].startswith("is_"):
                    new_cols.append(f'lctr_{col[0].replace("is_", "")}_count')
                elif col[1] == "max" and col[0].startswith("is_"):
                    new_cols.append(f'lctr_has_{col[0].replace("is_", "")}')
                else:
                    new_cols.append(f"lctr_{col[1]}_{col[0]}")
            else:
                new_cols.append(col)

        result.columns = new_cols

        return result

    def _empty_features(self) -> pd.DataFrame:
        """Return empty DataFrame with expected columns."""
        return pd.DataFrame(columns=["transcript_id"])


class UnknownRepeatProcessor(RepeatMaskerProcessor):
    """
    Process Unknown, Other, and Unclassified repeat classes.

    Features:
    - Count and total length
    - Gaps
    """

    UNKNOWN_CLASSES = {"Unknown": []}

    def extract_features(self) -> pd.DataFrame:
        """
        Extract Unknown repeat features at transcript level.

        Returns
        -------
        pd.DataFrame
            Transcript-level Unknown features.
        """
        logger.info("Extracting Unknown repeat features...")

        unknown_hits = self.rm_hits[
            self.rm_hits["hit_class"].isin(self.UNKNOWN_CLASSES.keys())
        ].copy()

        if unknown_hits.empty:
            logger.warning("No Unknown hits found!")
            return self._empty_features()

        logger.info(
            f"Processing {len(unknown_hits)} Unknown hits in "
            f"{unknown_hits['transcript_id'].nunique()} transcripts."
        )

        # Calculate basic metrics
        unknown_hits = self.calculate_hit_lengths(unknown_hits)
        unknown_hits = self.calculate_gaps_vectorized(unknown_hits)

        # Aggregate by element
        by_element = self._aggregate_by_element(unknown_hits)

        # Aggregate by transcript
        by_transcript = self._aggregate_by_transcript(by_element)

        # Calculate gap statistics
        gap_stats = self.calculate_gap_stats_vectorized(unknown_hits, "unknown")
        by_transcript = by_transcript.merge(gap_stats, on="transcript_id", how="left")

        return by_transcript

    def _aggregate_by_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Unknown fragments into elements."""
        grouped = df.groupby("id")

        agg_dict = {
            "transcript_id": "first",
            "hit_length": "sum",
        }

        result = grouped.agg(agg_dict).reset_index()
        result["num_fragments"] = grouped.size().values

        return result

    def _aggregate_by_transcript(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate elements into transcript-level Unknown features."""
        grouped = df.groupby("transcript_id", observed=True)

        agg_dict = {
            "hit_length": ["count", "sum", "mean"],
            "num_fragments": ["sum", "mean"],
        }

        result = grouped.agg(agg_dict).reset_index()

        # Flatten column names
        new_cols = ["transcript_id"]
        for col in result.columns[1:]:
            if isinstance(col, tuple):
                if col[1] == "count":
                    new_cols.append("unknown_count")
                elif col[1] == "sum" and col[0] == "hit_length":
                    new_cols.append("unknown_total_length")
                elif col[1] == "mean" and col[0] == "hit_length":
                    new_cols.append("unknown_mean_length")
                else:
                    new_cols.append(f"unknown_{col[1]}_{col[0]}")
            else:
                new_cols.append(col)

        result.columns = new_cols

        return result

    def _empty_features(self) -> pd.DataFrame:
        """Return empty DataFrame with expected columns."""
        return pd.DataFrame(columns=["transcript_id"])


class PseudogeneRepeatProcessor(RepeatMaskerProcessor):
    """
    Process Pseudogene repeat classes (rRNA, scRNA, snRNA, tRNA).

    Features:
    - Count and total length
    - Gaps
    - Divergence-based age classification
    - Fragmentation patterns
    - Family/subfamily diversity
    - Class-specific distributions
    """

    PSEUDO_CLASSES = {
        "rRNA": [],
        "scRNA": [],
        "snRNA": [],
        "tRNA": [],
    }

    def extract_features(self) -> pd.DataFrame:
        """
        Extract Pseudogene repeat features at transcript level.

        Returns
        -------
        pd.DataFrame
            Transcript-level Pseudogene features.
        """
        logger.info("Extracting Pseudogene repeat features...")

        pseudo_hits = self.rm_hits[
            self.rm_hits["hit_class"].isin(self.PSEUDO_CLASSES.keys())
        ].copy()

        if pseudo_hits.empty:
            logger.warning("No Pseudogene hits found!")
            return self._empty_features()

        logger.info(
            f"Processing {len(pseudo_hits)} Pseudogene hits in "
            f"{pseudo_hits['transcript_id'].nunique()} transcripts."
        )

        # Calculate basic metrics
        pseudo_hits = self.calculate_hit_lengths(pseudo_hits)
        pseudo_hits = self.calculate_gaps_vectorized(pseudo_hits)

        # Aggregate by element
        by_element = self._aggregate_by_element(pseudo_hits)

        # Aggregate by transcript
        by_transcript = self._aggregate_by_transcript(by_element)

        # Calculate gap statistics
        gap_stats = self.calculate_gap_stats_vectorized(pseudo_hits, "pseudogene")
        by_transcript = by_transcript.merge(gap_stats, on="transcript_id", how="left")

        return by_transcript

    def _aggregate_by_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Pseudogene fragments into elements."""
        grouped = df.groupby("id")

        agg_dict = {
            "transcript_id": "first",
            "hit_length": "sum",
            "hit_reference_coverage": "sum",
            "sw_score": "first",
            "divergence": "first",  # Meaningful for Pseudogenes
            "perc_del": "first",
            "perc_ins": "first",
            "strand": "first",
            "hit_name": "first",
            "hit_family": "first",
            "hit_class": "first",
        }

        result = grouped.agg(agg_dict).reset_index()
        result["num_fragments"] = grouped.size().values
        result["fragmented"] = result["num_fragments"] > 1

        # Add class boolean columns
        for pseudo_class in self.PSEUDO_CLASSES.keys():
            result[f"is_{pseudo_class.lower()}"] = result["hit_class"] == pseudo_class

        result["is_ancient"] = result["divergence"].astype(float) > 20

        return result

    def _aggregate_by_transcript(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate elements into transcript-level Pseudogene features."""
        grouped = df.groupby("transcript_id", observed=True)

        agg_dict = {
            # Divergence stats (age)
            "sw_score": ["min", "max", "mean"],
            "divergence": ["min", "max", "mean"],
            "perc_del": ["min", "max", "mean"],
            "perc_ins": ["min", "max", "mean"],
            # Length stats
            "hit_length": ["count", "sum", "min", "mean", "max"],
            "hit_reference_coverage": ["min", "mean", "max"],
            "num_fragments": ["sum", "mean", "max"],
            "fragmented": "sum",
            # Diversity
            "hit_name": "nunique",
            "hit_class": "nunique",
            "hit_family": "nunique",
        }

        # Add class-specific aggregations
        for pseudo_class in self.PSEUDO_CLASSES.keys():
            class_col = f"is_{pseudo_class.lower()}"
            agg_dict[class_col] = ["max", "sum"]

        # Age aggregation
        agg_dict["is_ancient"] = "sum"

        result = grouped.agg(agg_dict).reset_index()

        # Flatten multi-index column names
        new_cols = ["transcript_id"]
        for col in result.columns[1:]:
            if isinstance(col, tuple):
                if col[1] == "count":
                    new_cols.append(f"pseudo_{col[1]}")
                elif col[1] == "sum" and col[0].startswith("is_"):
                    # is_line sum → line_count
                    new_cols.append(f'pseudo_{col[0].replace("is_", "")}_count')
                elif col[1] == "max" and col[0].startswith("is_"):
                    # is_line max → has_line
                    new_cols.append(f'pseudo_has_{col[0].replace("is_", "")}')
                elif col[0] == "hit_name" and col[1] == "nunique":
                    new_cols.append("pseudo_unique_subfamilies")
                elif col[0] == "hit_class" and col[1] == "nunique":
                    new_cols.append("pseudo_unique_classes")
                elif col[0] == "hit_family" and col[1] == "nunique":
                    new_cols.append("pseudo_unique_families")
                else:
                    new_cols.append(f"pseudo_{col[1]}_{col[0]}")
            else:
                new_cols.append(col)

        result.columns = new_cols

        return result

    def _empty_features(self) -> pd.DataFrame:
        """Return empty DataFrame with expected columns."""
        return pd.DataFrame(columns=["transcript_id"])


class GlobalRepeatProcessor(RepeatMaskerProcessor):
    """
    Calculate features across ALL repeat types.

    Global features:
    - Total repeat content (all RM hits)
    - Cross-class gap statistics
    - Overall repeat density
    """

    def extract_features(self) -> pd.DataFrame:
        """
        Extract global repeat features (all RM hits together).

        Returns
        -------
        pd.DataFrame
            Transcript-level global features with columns:
            - Total RM count and length
            - Global gap statistics (between ANY consecutive hits)
            - Overall repeat density
        """
        logger.info("Extracting global repeat features...")

        if self.rm_hits.empty:
            logger.warning("No RM hits found!")
            return pd.DataFrame(columns=["transcript_id"])

        # Calculate basic metrics on ALL hits
        all_hits = self.calculate_hit_lengths(self.rm_hits)
        all_hits = self.calculate_gaps_vectorized(all_hits)

        # Global coverage
        grouped = all_hits.groupby("transcript_id", observed=True)

        coverage = grouped.agg(
            {
                "hit_length": ["count", "sum"],
            }
        ).reset_index()
        coverage.columns = [
            "transcript_id",
            "global_rm_count",
            "global_rm_total_length",
        ]

        # Global gap stats (between ANY consecutive RM hits)
        gap_stats = self.calculate_gap_stats_vectorized(all_hits, "global")
        global_features = coverage.merge(gap_stats, on="transcript_id", how="left")

        return global_features


# =============================================================================
# METADATA INTEGRATION
# =============================================================================


class TranscriptAnnotator:
    """Add external metadata (biotypes, coding classes) to features."""

    @staticmethod
    def add_biotypes(features: pd.DataFrame, biotypes: pd.DataFrame) -> pd.DataFrame:
        """
        Add biotype annotations.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with transcript_id
        biotypes : pd.DataFrame
            Biotype DataFrame with transcript_id and transcript_type

        Returns
        -------
        pd.DataFrame
            Features with transcript_type column added
        """
        if biotypes is None:
            return features

        logger.info("Adding biotype annotations...")
        biotype_map = biotypes.set_index("transcript_id")[["transcript_type"]]

        features = features.merge(
            biotype_map, left_on="transcript_id", right_index=True, how="left"
        )

        return features

    @staticmethod
    def add_coding_classes(
        features: pd.DataFrame, pc_ids: Set[str], lnc_ids: Set[str]
    ) -> pd.DataFrame:
        """
        Add coding class labels (coding, lncRNA, other).

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with transcript_id
        pc_ids : Set[str]
            Protein-coding transcript IDs
        lnc_ids : Set[str]
            lncRNA transcript IDs

        Returns
        -------
        pd.DataFrame
            Features with coding_class column added
        """
        if not pc_ids and not lnc_ids:
            return features

        logger.info("Adding coding class labels...")

        # Ensure no overlap
        overlap = pc_ids.intersection(lnc_ids)
        if overlap:
            logger.warning(
                f"PC and lncRNA ID sets overlap ({len(overlap)} transcripts)!"
            )

        is_coding = features["transcript_id"].isin(pc_ids)
        is_lncrna = features["transcript_id"].isin(lnc_ids)

        features["coding_class"] = np.select(
            [is_coding, is_lncrna], ["coding", "lncRNA"], default="other"
        )

        logger.info(
            f"Coding class distribution:\n{features['coding_class'].value_counts()}"
        )

        return features


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


class TEFeatureExtractor:
    """
    Main orchestrator for TE feature extraction pipeline.

    Coordinates:
    1. Data loading (RepeatMasker + transcripts + metadata)
    2. Overlap filtering
    3. Feature extraction by repeat type (TE, LC/TR, global)
    4. Metadata integration
    5. Relative metric calculation
    """

    def __init__(
        self,
        repeatmasker_file: str,
        transcripts_bed: str,
        biotypes_file: Optional[str] = None,
        pc_ids_file: Optional[str] = None,
        lnc_ids_file: Optional[str] = None,
        lengths_file: Optional[str] = None,
        output_prefix: str = "",
    ):
        """
        Initialize TE feature extractor.

        Parameters
        ----------
        repeatmasker_file : str
            RepeatMasker .out file
        transcripts_bed : str
            Transcript coordinates BED file
        biotypes_file : str, optional
            TSV file with transcript biotypes
        pc_ids_file : str, optional
            File with protein-coding transcript IDs (one per line)
        lnc_ids_file : str, optional
            File with lncRNA transcript IDs (one per line)
        lengths_file : str, optional
            TSV file with pre-calculated spliced transcript lengths
            (transcript_id\tlength), as produced by extract_transcript_lengths.
            When provided, overrides the genomic-span lengths derived from the
            BED coordinates, ensuring TE coverage percentages are relative to
            the actual spliced sequence length.
        output_prefix : str
            Output file prefix
        """
        self.repeatmasker_file = repeatmasker_file
        self.transcripts_bed = transcripts_bed
        self.biotypes_file = biotypes_file
        self.pc_ids_file = pc_ids_file
        self.lnc_ids_file = lnc_ids_file
        self.lengths_file = lengths_file
        self.output_prefix = output_prefix

        # Will be populated during execution
        self.rm_hits = None
        self.transcripts = None
        self.biotypes = None
        self.pc_ids = set()
        self.lnc_ids = set()
        self.features = None

    def run(self):
        """Execute complete feature extraction pipeline."""
        logger.info("=" * 80)
        logger.info("TE Feature Extractor - Starting (Refactored Version)")
        logger.info("=" * 80)

        # 1. Load data
        logger.info("[Step 1/4] Loading data...")
        self.load_data()

        # 2. Filter overlaps
        logger.info("[Step 2/4] Filtering overlaps...")
        self.filter_overlaps()

        # 3. Extract features by repeat type
        logger.info("[Step 3/4] Extracting features by repeat type...")
        self.extract_features()

        # 4. Save results
        logger.info("[Step 4/4] Saving results...")
        self.save_features()

        logger.info("=" * 80)
        logger.info("Feature extraction complete!")
        logger.info("=" * 80)

    def load_data(self):
        """Load all input data."""
        logger.info("Loading data...")

        # Load RepeatMasker hits (.out format only)
        self.rm_hits = load_repeatmasker_out(self.repeatmasker_file)
        logger.info(f"Loaded {len(self.rm_hits)} RepeatMasker annotations")

        # Optimize memory: convert to categorical
        self.rm_hits["hit_class"] = self.rm_hits["hit_class"].astype("category")
        self.rm_hits["hit_family"] = self.rm_hits["hit_family"].astype("category")
        self.rm_hits["transcript_id"] = self.rm_hits["transcript_id"].astype("category")

        # Load transcripts
        logger.info(f"Reading transcripts from {self.transcripts_bed}")
        self.transcripts = pd.read_csv(
            self.transcripts_bed,
            sep="\t",
            header=None,
            names=["chrom", "start", "end", "transcript_id", "score", "strand"],
            usecols=[0, 1, 2, 3, 4, 5],
        )

        # Override genomic-span lengths with GTF-derived spliced lengths if provided
        if self.lengths_file:
            logger.info(
                f"Loading pre-calculated spliced lengths from {self.lengths_file}"
            )
            gtf_lengths = pd.read_csv(
                self.lengths_file,
                sep="\t",
                header=None,
                names=["transcript_id", "length"],
                index_col=0,
            )
            gtf_length_map = gtf_lengths["length"].to_dict()
            # Map with version-stripped fallback
            mapped = self.transcripts["transcript_id"].map(gtf_length_map)
            if mapped.isna().any():
                stripped_map = {k.split(".")[0]: v for k, v in gtf_length_map.items()}
                mapped = mapped.fillna(
                    self.transcripts["transcript_id"]
                    .str.split(".")
                    .str[0]
                    .map(stripped_map)
                )
            overridden = mapped.notna().sum()
            logger.info(
                f"Overriding lengths for {overridden}/{len(self.transcripts)} transcripts "
                "with GTF-derived spliced lengths"
            )
            self.transcripts["length"] = mapped
        else:
            # TODO: properly configure object so that lengths_file is required
            raise ValueError(
                "lengths_file is required to ensure accurate TE coverage calculations relative to spliced transcript lengths. Please provide the output of extract_transcript_lengths."
            )

        # Map transcript lengths to RM hits
        tx_length_map = self.transcripts.set_index("transcript_id")["length"].to_dict()
        tx_lengths = self.rm_hits["transcript_id"].map(tx_length_map)

        logger.info(f"Loaded {len(self.transcripts)} transcripts")

        # Workaround to fix when RM hits have versioned IDs but transcripts do not (TODO: Fix this in upstream rule)
        if tx_lengths.isna().any():
            missing_count = tx_lengths.isna().sum()
            logger.warning(
                f"{missing_count} RM hits have no matching transcript length!"
            )
            logger.warning("Trying to map without version suffix...")

            # RM hits have versioned IDs, but transcripts may not
            tx_lengths = (
                self.rm_hits["transcript_id"]
                .astype(str)
                .str.split(".")
                .str[0]
                .map(tx_length_map)
            )
            mapped_count = tx_lengths.notna().sum()
            logger.info(f"Mapped additional {mapped_count} hits without version suffix")

            # # New df with no_version_id as index
            # no_version_df = self.transcripts.set_index('transcript_id').copy()
            # no_version_df.index = no_version_df.index.str.split('.').str[0]
            # tx_length_map_no_version = no_version_df['length'].to_dict()

        logger.info("All RM hits successfully mapped to transcript lengths")
        self.rm_hits["transcript_length"] = tx_lengths

        # Filter to hits with matching transcripts
        self.rm_hits = self.rm_hits[self.rm_hits["transcript_length"].notna()].copy()
        logger.info(f"Retained {len(self.rm_hits)} hits with matching transcripts")

        # Load optional metadata
        if self.biotypes_file:
            logger.info(f"Reading biotypes from {self.biotypes_file}")
            self.biotypes = pd.read_csv(self.biotypes_file, sep="\t")
            logger.info(f"Loaded biotypes for {len(self.biotypes)} transcripts")

        if self.pc_ids_file:
            logger.info(f"Reading PC transcript IDs from {self.pc_ids_file}")
            with open(self.pc_ids_file, "r") as f:
                self.pc_ids = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.pc_ids)} protein-coding IDs")

        if self.lnc_ids_file:
            logger.info(f"Reading lncRNA transcript IDs from {self.lnc_ids_file}")
            with open(self.lnc_ids_file, "r") as f:
                self.lnc_ids = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.lnc_ids)} lncRNA IDs")

    def filter_overlaps(self):
        """Detect and filter overlapping hits."""
        logger.info("Detecting overlapping hits...")

        # Find overlaps
        overlaps = OverlapFilter.find_overlapping_hits(self.rm_hits)

        # Filter
        self.rm_hits = OverlapFilter.filter_overlapping_hits(self.rm_hits, overlaps)

        # Sort for gap calculation
        self.rm_hits = self.rm_hits.sort_values(["transcript_id", "start"]).reset_index(
            drop=True
        )

    def extract_features(self):
        """Extract features by repeat type and merge."""
        logger.info("Extracting features by repeat type...")

        # Initialize processors
        te_proc = TEProcessor(self.rm_hits, self.transcripts)
        lctr_proc = LowComplexityTandemProcessor(self.rm_hits, self.transcripts)
        pseudo_proc = PseudogeneRepeatProcessor(self.rm_hits, self.transcripts)
        unknown_proc = UnknownRepeatProcessor(self.rm_hits, self.transcripts)
        global_proc = GlobalRepeatProcessor(self.rm_hits, self.transcripts)

        # Extract features
        logger.info("  - Extracting TE features...")
        te_features = te_proc.extract_features()

        logger.info("  - Extracting Low Complexity/Tandem Repeat features...")
        lctr_features = lctr_proc.extract_features()

        logger.info("  - Extracting Pseudogene features...")
        pseudo_features = pseudo_proc.extract_features()

        logger.info("  - Extracting Unknown repeat features...")
        unknown_features = unknown_proc.extract_features()

        logger.info("  - Extracting Global repeat features...")
        global_features = global_proc.extract_features()

        # Start with transcript base
        self.features = self.transcripts[["transcript_id", "length"]].copy()
        self.features = self.features.rename(columns={"length": "transcript_length"})

        # Merge feature sets (left join to preserve all transcripts)
        logger.info("  - Merging feature sets...")
        self.features = self.features.merge(te_features, on="transcript_id", how="left")
        self.features = self.features.merge(
            lctr_features, on="transcript_id", how="left"
        )
        self.features = self.features.merge(
            pseudo_features, on="transcript_id", how="left"
        )
        self.features = self.features.merge(
            unknown_features, on="transcript_id", how="left"
        )
        self.features = self.features.merge(
            global_features, on="transcript_id", how="left"
        )

        # Fill NaNs
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        self.features[numeric_cols] = self.features[numeric_cols].fillna(0)

        # Calculate relative metrics
        logger.info("  - Calculating relative metrics...")
        self._calculate_relative_metrics()

        # Add metadata
        logger.info("  - Adding metadata and annotations...")
        annotator = TranscriptAnnotator()
        self.features = annotator.add_biotypes(self.features, self.biotypes)
        self.features = annotator.add_coding_classes(
            self.features, self.pc_ids, self.lnc_ids
        )

        logger.info(
            f"Extracted {len(self.features.columns)} features for {len(self.features)} transcripts"
        )

    def _calculate_relative_metrics(self):
        """Calculate percentage and per-kb metrics."""
        # Percentage metrics
        pct_cols = [
            "te_sum_hit_length",
            "te_min_hit_length",
            "te_mean_hit_length",
            "te_max_hit_length",
            "te_gaps_mean",
            "te_gaps_median",
            "te_gaps_max",
            "te_gaps_min",
            "lctr_total_length",
            "lctr_mean_length",
            "unknown_total_length",
            "unknown_mean_length",
            "global_rm_total_length",
            "global_gaps_mean",
            "global_gaps_median",
        ]

        for col in pct_cols:
            if col in self.features.columns:
                self.features[f"{col}_pct"] = (
                    self.features[col] / self.features["transcript_length"] * 100
                )

        # Per kb metrics
        perkb_cols = ["te_count", "lctr_count", "unknown_count", "global_rm_count"]
        perkb_cols += [c for c in self.features.columns if c.endswith("_count")]

        for col in perkb_cols:
            if col in self.features.columns:
                self.features[f"{col}_per_kb"] = (
                    self.features[col] / self.features["transcript_length"] * 1000
                )

    def save_features(self):
        """Save features and summary."""
        output_file = f"{self.output_prefix}_te_features.csv"
        logger.info(f"Saving features to {output_file}")
        self.features.to_csv(output_file, index=False)
        logger.info(f"Saved {len(self.features)} transcripts")

        summary_file = f"{self.output_prefix}_te_summary.txt"
        logger.info(f"Saving summary to {summary_file}")

        with open(summary_file, "w") as f:
            f.write("TE Feature Extraction Summary (Refactored)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total transcripts: {len(self.features)}\n")

            if "te_count" in self.features.columns:
                has_te = (self.features["te_count"] > 0).sum()
                f.write(f"Transcripts with TEs: {has_te}\n")
                f.write(
                    f"Percentage with TEs: {has_te / len(self.features) * 100:.2f}%\n\n"
                )

            if "transcript_type" in self.features.columns:
                f.write("Biotype distribution:\n")
                for biotype, count in (
                    self.features["transcript_type"].value_counts().items()
                ):
                    f.write(f"  {biotype}: {count}\n")
                f.write("\n")

            if "coding_class" in self.features.columns:
                f.write("Coding class distribution:\n")
                for cls, count in self.features["coding_class"].value_counts().items():
                    f.write(f"  {cls}: {count}\n")
                f.write("\n")

            f.write("Feature statistics:\n")
            f.write(self.features.describe().to_string())

        logger.info(f"Summary saved to {summary_file}")


# =============================================================================
# BACKWARD COMPATIBILITY - Original Class Name
# =============================================================================


class TEFeatureExtractor_Flexible(TEFeatureExtractor):
    """
    Backward-compatible wrapper for original class name.

    Preserves the original API while using the refactored implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self):
        """Backward-compatible load_data method."""
        super().load_data()

    def find_overlapping_hits(self, **kwargs):
        """Backward-compatible find_overlapping_hits method."""
        overlaps = OverlapFilter.find_overlapping_hits(self.rm_hits, **kwargs)
        return overlaps

    def filter_overlapping_hits(self):
        """Backward-compatible filter_overlapping_hits method."""
        overlaps = OverlapFilter.find_overlapping_hits(self.rm_hits)
        self.rm_hits = OverlapFilter.filter_overlapping_hits(self.rm_hits, overlaps)
        self.rm_hits = self.rm_hits.sort_values(["transcript_id", "start"]).reset_index(
            drop=True
        )

    def extract_basic_features(self) -> pd.DataFrame:
        """Backward-compatible extract_basic_features method."""
        self.extract_features()
        return self.features

    def save_features(self):
        """Backward-compatible save_features method."""
        super().save_features()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract TE features with biotype tracking (refactored modular version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --repeatmasker data.out --transcripts tx.bed --output-prefix out

  # With biotype information
  %(prog)s --repeatmasker data.out --transcripts tx.bed --biotypes bio.tsv --output-prefix out

  # With coding class labels
  %(prog)s --repeatmasker data.out --transcripts tx.bed --pc-ids pc.txt --lnc-ids lnc.txt --output-prefix out
        """,
    )
    parser.add_argument("--repeatmasker", required=True, help="RepeatMasker .out file")
    parser.add_argument(
        "--transcripts", required=True, help="Transcript coordinates BED file"
    )
    parser.add_argument(
        "--biotypes",
        required=False,
        default=None,
        help="File with transcript biotype information",
    )
    parser.add_argument(
        "--pc-ids",
        required=False,
        default=None,
        help="File with protein-coding transcript IDs (one per line)",
    )
    parser.add_argument(
        "--lnc-ids",
        required=False,
        default=None,
        help="File with lncRNA transcript IDs (one per line)",
    )
    parser.add_argument(
        "--lengths",
        required=False,
        default=None,
        help="TSV file with pre-calculated spliced transcript lengths "
        "(transcript_id<TAB>length), produced by extract_transcript_lengths",
    )
    parser.add_argument("--output-prefix", required=True, help="Output file prefix")

    args = parser.parse_args()

    extractor = TEFeatureExtractor(
        repeatmasker_file=args.repeatmasker,
        transcripts_bed=args.transcripts,
        biotypes_file=args.biotypes,
        pc_ids_file=args.pc_ids,
        lnc_ids_file=args.lnc_ids,
        lengths_file=args.lengths,
        output_prefix=args.output_prefix,
    )

    extractor.run()


if __name__ == "__main__":
    main()
