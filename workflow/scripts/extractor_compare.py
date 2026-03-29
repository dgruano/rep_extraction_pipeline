#!/usr/bin/env python3
"""
TE Feature Extractor - Optimized Version
========================================
Extract TE features with biotype tracking from full transcript database.
OPTIMIZED: Vectorized operations, efficient groupby, reduced memory usage
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

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
        if current % max(1, total // 10) == 0 or current == total:  # Log every 10%
            pct = (current / total) * 100 if total > 0 else 0
            logger.info(f"{desc}: {current}/{total} ({pct:.1f}%)")


class TEFeatureExtractor_Flexible:
    """Extract TE features with comprehensive biotype tracking."""

    def __init__(
        self,
        repeatmasker_file: str,
        transcripts_bed: str,
        biotypes_file: Optional[str] = None,
        pc_ids_file: Optional[str] = None,
        lnc_ids_file: Optional[str] = None,
        output_prefix: str = "",
        input_format: str = "auto",
    ):
        """
        Initialize flexible TE feature extractor.

        Parameters:
        -----------
        repeatmasker_file : str
            RepeatMasker output file (GFF3 or .out format)
        transcripts_bed : str, optional
            Transcript coordinates BED file (not needed for .out format)
        biotypes_file : str, optional
            File with transcript biotype information
        output_prefix : str
            Output file prefix
        input_format : str
            Input format: 'gff', 'out', or 'auto' (default)
        """
        self.repeatmasker_file = repeatmasker_file
        self.input_format = input_format
        self.transcripts_bed = transcripts_bed
        self.biotypes_file = biotypes_file
        self.pc_ids_file = pc_ids_file
        self.lnc_ids_file = lnc_ids_file
        self.output_prefix = output_prefix

        self.transcripts = None
        self.biotypes = None
        self.tes = None
        self.te_overlaps = None
        self.features = None

        # TE classification - pre-compile for faster lookups

        # Repetitive Element classes:
        """
        Transposable_Element: ['SINE', 'DNA', 'LTR', 'RC', 'LINE', 'Retroposon', 'PLE']
        Pseudogene: ['snRNA', 'rRNA', 'scRNA', 'tRNA', 'RNA']
        Satellite: ['Satellite']
        Unknown: ['Unknown']
        """
        # Transposable element classes and families:
        """
        ('DNA', ['TcMar-Tigger', 'hAT-Charlie', 'hAT-Tip100', 'hAT-Blackjack', 'TcMar-Mariner', 'TcMar-Tc2', 'hAT-Ac', 'hAT', 'PiggyBac', 'MULE-MuDR', 'hAT-Tag1', 'TcMar-Tc1', 'PIF-Harbinger', 'Kolobok', 'Merlin', 'Crypton', 'Crypton-A', 'TcMar'])
        ('LINE', ['L2', 'RTE-BovB', 'L1', 'CR1', 'RTE-X', 'Dong-R4', 'I-Jockey', 'L1-Tx1'])
        ('LTR', ['ERVL', 'ERVL-MaLR', 'ERV1', 'ERVK', 'DIRS', 'Gypsy'])
        ('Low_complexity', [])
        ('PLE', [])
        ('RC', ['Helitron'])
        ('Retroposon', ['SVA'])
        ('SINE', ['Alu', 'MIR', 'tRNA', 'tRNA-RTE', '5S-Deu-L2', 'tRNA-Deu'])
        """

        self.te_classes = {
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

        # Pre-compile keywords for faster matching
        self._class_keywords = {}
        for class_name, keywords in self.te_classes.items():
            self._class_keywords[class_name] = [kw.upper() for kw in keywords]

        self.young_subfamilies = {
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

    def load_data(self):
        """Load all input data."""
        logger.info("Loading data...")

        # Auto-detect format if needed
        if self.input_format == "auto":
            if self.repeatmasker_file.endswith(
                ".gff"
            ) or self.repeatmasker_file.endswith(".gff3"):
                self.input_format = "gff"
            elif self.repeatmasker_file.endswith(".out"):
                self.input_format = "out"
            else:
                # Check file content
                with open(self.repeatmasker_file, "r") as f:
                    first_line = f.readline()
                    if first_line.startswith("##gff-version"):
                        self.input_format = "gff"
                    elif "SW" in first_line and "score" in first_line:
                        self.input_format = "out"
                    else:
                        raise ValueError(
                            "Cannot auto-detect format. Please specify --format"
                        )

        logger.info(f"Detected format: {self.input_format}")

        # Load TEs
        logger.info(f"Reading TEs from {self.repeatmasker_file}")
        if self.input_format == "gff":
            self.tes = self._load_gff3()
        else:
            self.tes = self._load_out()

        logger.info(f"Loaded {len(self.tes)} TE annotations")

        # Extract transcript info from TEs (for .out format)
        if self.transcripts_bed is None:
            logger.info("Extracting transcript information from TE data...")
            self.transcripts = self._extract_transcripts_from_tes()
        else:
            # Load transcripts from BED
            logger.info(f"Reading transcripts from {self.transcripts_bed}")
            self.transcripts = pd.read_csv(
                self.transcripts_bed,
                sep="\t",
                header=None,
                names=["chrom", "start", "end", "transcript_id", "score", "strand"],
                usecols=[0, 1, 2, 3, 4, 5],  # Only read needed columns
            )
            self.transcripts["length"] = (
                self.transcripts["end"] - self.transcripts["start"]
            )
            self.transcripts["transcript_id_base"] = (
                self.transcripts["transcript_id"].str.split(".").str[0]
            )

        logger.info(f"Loaded {len(self.transcripts)} transcripts")

        # Load biotypes if provided
        if self.biotypes_file:
            logger.info(f"Reading biotypes from {self.biotypes_file}")
            self.biotypes = pd.read_csv(self.biotypes_file, sep="\t")
            logger.info(f"Loaded biotypes for {len(self.biotypes)} transcripts")

        # Load PC transcript IDs if provided
        if self.pc_ids_file:
            logger.info(f"Reading PC transcript IDs from {self.pc_ids_file}")
            with open(self.pc_ids_file, "r") as f:
                self.pc_ids = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.pc_ids)} protein-coding transcript IDs")

        # Load lncRNA transcript IDs if provided
        if self.lnc_ids_file:
            logger.info(f"Reading lncRNA transcript IDs from {self.lnc_ids_file}")
            with open(self.lnc_ids_file, "r") as f:
                self.lnc_ids = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.lnc_ids)} lncRNA transcript IDs")

    def _load_gff3(self) -> pd.DataFrame:
        """Load RepeatMasker GFF3."""
        gff_cols = [
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attributes",
        ]

        tes = pd.read_csv(
            self.repeatmasker_file,
            sep="\t",
            comment="#",
            header=None,
            names=gff_cols,
            usecols=[
                "seqname",
                "start",
                "end",
                "attributes",
            ],  # Only read needed columns
        )

        # Parse attributes - vectorized
        tes["te_name"] = tes["attributes"].str.extract(r"Target=([^\s]+)", expand=False)
        tes["te_family"] = tes["attributes"].str.extract(
            r"Family=([^;]+)", expand=False
        )
        tes["te_class"] = tes["attributes"].str.extract(r"Class=([^;]+)", expand=False)
        tes["divergence"] = pd.to_numeric(
            tes["attributes"].str.extract(r"divergence=([\d.]+)", expand=False),
            errors="coerce",
        ).astype("Float64")
        tes["te_length"] = tes["end"] - tes["start"]

        # Normalize transcript ID
        tes["transcript_id"] = tes["seqname"].str.split(".").str[0]

        # Drop attributes column to save memory
        tes = tes.drop(columns=["seqname", "attributes"])

        return tes

    def _load_out(self) -> pd.DataFrame:
        """Load RepeatMasker .out file."""
        # RepeatMasker .out format has 3 header lines, then data
        # Read the file, skipping header lines
        with open(self.repeatmasker_file, "r") as f:
            lines = f.readlines()

        # Find where data starts (after the header lines)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(" ") and "SW" not in line:
                data_start = i
                break
            if "ID" in line:  # Last header line
                data_start = i + 1
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

        # Read data with fixed-width or whitespace-separated
        # The .out format is whitespace-separated
        tes = pd.read_csv(
            self.repeatmasker_file,
            sep=r"\s+",
            skiprows=data_start,
            names=col_names,
            engine="python",
        )

        # Clean up columns
        tes["query_sequence"] = tes["query_sequence"].astype(str)
        tes["query_begin"] = pd.to_numeric(tes["query_begin"], errors="coerce").astype(
            "Int64"
        )
        tes["query_end"] = pd.to_numeric(tes["query_end"], errors="coerce").astype(
            "Int64"
        )

        # Parse strand and repeat info
        # Strand is indicated by 'C' for complement or '+' for forward
        tes["strand"] = tes["strand"].replace({"C": "-", "+": "+"})

        # Split repeat_class_family into class and family
        class_family_split = tes["repeat_class_family"].str.split("/", expand=True)
        tes["te_class"] = (
            class_family_split[0] if len(class_family_split.columns) > 0 else None
        )
        tes["te_family"] = (
            class_family_split[1] if len(class_family_split.columns) > 1 else None
        )

        # Rename columns to match expected format
        tes = tes.rename(
            columns={
                "query_sequence": "transcript_id",
                "query_begin": "start",
                "query_end": "end",
                "repeat_name": "te_name",
                "perc_div": "divergence",
            }
        )

        # Calculate TE length
        tes["te_length"] = tes["end"] - tes["start"]

        # Normalize transcript ID (remove version if present)
        tes["transcript_id"] = tes["transcript_id"].str.split(".").str[0]

        # Keep only necessary columns
        tes = tes[
            [
                "transcript_id",
                "start",
                "end",
                "te_name",
                "te_family",
                "te_class",
                "divergence",
                "te_length",
                "strand",
            ]
        ]

        return tes

    def _extract_transcripts_from_tes(self) -> pd.DataFrame:
        """Extract transcript information from TE data."""
        # Get unique transcripts and calculate their lengths from TE positions
        transcript_info = (
            self.tes.groupby("transcript_id")
            .agg(
                {
                    "end": "max",  # Maximum position as transcript end
                    "start": "min",  # Minimum position as transcript start
                }
            )
            .reset_index()
        )

        transcript_info["length"] = transcript_info["end"] - transcript_info["start"]
        transcript_info["transcript_id_base"] = (
            transcript_info["transcript_id"].str.split(".").str[0]
        )

        return transcript_info

    def calculate_overlaps(self):
        """Calculate TE-transcript overlaps using vectorized operations."""
        logger.info("Calculating overlaps...")

        if len(self.tes) == 0:
            self.te_overlaps = pd.DataFrame()
            logger.info("No TEs found")
            return

        # Create a lookup dictionary for transcript lengths
        tx_length_map = self.transcripts.set_index("transcript_id")["length"].to_dict()

        # Vectorized overlap calculation - much faster than looping
        self.tes["overlap_length"] = self.tes["te_length"]
        self.tes["transcript_id_full"] = self.tes["transcript_id"]
        self.tes["transcript_length"] = self.tes["transcript_id"].map(tx_length_map)

        # Filter to only TEs that have matching transcripts
        self.te_overlaps = self.tes[self.tes["transcript_length"].notna()].copy()

        logger.info(f"Found {len(self.te_overlaps)} TE overlaps")

    def classify_te_class_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Classify TE to major class using vectorized operations."""
        # Combine family and class info
        search_str = (
            df["te_family"].fillna("") + " " + df["te_class"].fillna("")
        ).str.upper()

        # Initialize with 'Unknown'
        result = pd.Series("Unknown", index=df.index)

        # Check each class
        for class_name, keywords in self._class_keywords.items():
            # Create mask for this class
            mask = pd.Series(False, index=df.index)
            for keyword in keywords:
                mask |= search_str.str.contains(keyword, regex=False, na=False)
            result[mask] = class_name

        # Anything not Unknown but also not in main classes
        result[(result == "Unknown") & (search_str != " ")] = "Other"

        return result

    def extract_possible_te_classes(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        This function reads the dataframe and gets all the unique TE classes and families
        associated with columns 'te_class' and 'te_family'. It the associates each family
        with a major class based on the dataframe data.

        Finally, it returns a dictionary mapping each TE family to its major class.
        te_classes =  ['SINE', 'DNA', 'LTR', 'RC', 'LINE', 'Retroposon', 'PLE']
        pseudogene_classes = ['snRNA', 'rRNA', 'scRNA', 'tRNA', 'RNA']
        satellite_classes = ['Satellite']
        unknown_classes = ['Unknown']

        ('DNA', ['TcMar-Tigger', 'hAT-Charlie', 'hAT-Tip100', 'hAT-Blackjack', 'TcMar-Mariner', 'TcMar-Tc2', 'hAT-Ac', 'hAT', 'PiggyBac', 'MULE-MuDR', 'hAT-Tag1', 'TcMar-Tc1', 'PIF-Harbinger', 'Kolobok', 'Merlin', 'Crypton', 'Crypton-A', 'TcMar'])
        ('LINE', ['L2', 'RTE-BovB', 'L1', 'CR1', 'RTE-X', 'Dong-R4', 'I-Jockey', 'L1-Tx1'])
        ('LTR', ['ERVL', 'ERVL-MaLR', 'ERV1', 'ERVK', 'DIRS', 'Gypsy'])
        ('Low_complexity', [])
        ('PLE', [])
        ('RC', ['Helitron'])
        ('Retroposon', ['SVA'])
        ('SINE', ['Alu', 'MIR', 'tRNA', 'tRNA-RTE', '5S-Deu-L2', 'tRNA-Deu'])
        ('Satellite', ['centr', 'subtelo', 'acro'])
        ('Simple_repeat', [])
        ('Unknown', [])
        ('begin', [])
        ('rRNA', [])
        ('scRNA', [])
        ('snRNA', [])
        ('srpRNA', [])
        ('tRNA', [])
        """
        pass
        return

    def extract_basic_features(self) -> pd.DataFrame:
        """Extract all basic features using vectorized groupby operations."""
        logger.info("Extracting basic features...")

        # Classify all TEs at once - vectorized
        if len(self.te_overlaps) > 0:
            # Pre-calculate young TE indicator
            self.te_overlaps["is_young"] = self.te_overlaps["te_name"].isin(
                self.young_subfamilies
            )
            self.te_overlaps["is_ancient"] = self.te_overlaps["divergence"] > 20

            # Pre-calculate family matches
            for te_class, te_class_families in self.te_classes.items():
                logger.info(f"Classifying {te_class} hits...")
                # Create bool column for class
                self.te_overlaps[f"is_{te_class.lower()}"] = (
                    self.te_overlaps["te_class"] == te_class
                )
                # Create bool columns for each family in this class
                for family in te_class_families:
                    self.te_overlaps[f"is_{family.lower()}"] = (
                        self.te_overlaps["te_family"] == family
                    )

        logger.info(self.te_overlaps.head())
        # Use groupby aggregation - much faster than looping
        logger.info("Aggregating features by transcript...")

        if len(self.te_overlaps) > 0:
            grouped = self.te_overlaps.groupby("transcript_id")

            # Dynamically build aggregation dictionary for is_ columns
            agg_dict = {
                "te_name": ["count", "nunique"],
                "te_family": "nunique",
                "overlap_length": "sum",
                "te_length": ["mean", "max"],
                "divergence": "mean",
            }

            agg_names = [
                "transcript_id",
                "te_count",
                "unique_subfamilies",
                "unique_families",
                "te_coverage_bp",
                "mean_te_length",
                "max_te_span",
                "mean_te_divergence",
            ]

            # Add all is_ columns with sum aggregation
            is_cols = [col for col in self.te_overlaps.columns if col.startswith("is_")]
            for col in is_cols:
                agg_dict[col] = "sum"
                agg_names.append(col.replace("is_", "") + "_count")

            logger.debug(f"New column names: {agg_names}")

            # Aggregate all features at once
            agg_features = grouped.agg(agg_dict).reset_index()
            agg_features.columns = agg_names

            # Aggregate all features at once

            """
            agg_features = grouped.agg({
                'te_name': ['count', 'nunique'],
                'te_family': 'nunique',
                'overlap_length': 'sum',
                'te_length': ['mean', 'max'],
                'divergence': 'mean',
                'is_young': 'sum',
                'is_ancient': 'sum',
                'is_l1': 'sum',
                'is_alu': 'sum',
                'is_sva': 'sum',
                'is_herv': 'sum',
            }).reset_index()
            """

            # Flatten column names
            """
            agg_features.columns = [
                'transcript_id', 'te_count', 'unique_subfamilies', 'unique_families',
                'te_coverage_bp', 'mean_te_length', 'max_te_span', 'mean_te_divergence',
                'young_te_count', 'ancient_te_count', 'l1_count', 'alu_count',
                'sva_count', 'herv_count'
            ]
            """
            """
            # Count by major class
            class_counts = grouped['te_class'].value_counts().unstack(fill_value=0).reset_index()
            class_counts.columns.name = None

            # Merge class counts
            agg_features = agg_features.merge(class_counts, on='transcript_id', how='left')

            # Rename and fill class columns
            for class_name in ['LINE', 'SINE', 'LTR', 'DNA']:
                col_name = f'{class_name.lower()}_count'
                if class_name in agg_features.columns:
                    agg_features = agg_features.rename(columns={class_name: col_name})
                else:
                    agg_features[col_name] = 0
            """
        else:
            agg_features = pd.DataFrame(columns=["transcript_id"])

        # Merge with all transcripts
        self.features = self.transcripts[["transcript_id", "length"]].merge(
            agg_features, on="transcript_id", how="left"
        )

        # Rename and calculate additional columns
        self.features = self.features.rename(columns={"length": "transcript_length"})
        self.features["te_present"] = (~self.features["te_count"].isna()).astype(int)

        # Fill NaN values with 0 for numeric columns
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        self.features[numeric_cols] = self.features[numeric_cols].fillna(0)

        # Calculate percentage coverage
        self.features["te_coverage_pct"] = (
            self.features["te_coverage_bp"] / self.features["transcript_length"] * 100
        ).fillna(0)

        # Add biotype information if available
        if self.biotypes is not None:
            logger.info("Merging biotype information...")
            biotype_map = self.biotypes.set_index("transcript_id_base")[
                ["transcript_type"]
            ]
            self.features = self.features.merge(
                biotype_map, left_on="transcript_id", right_index=True, how="left"
            )

        # Add coding class if available
        if self.pc_ids_file and self.lnc_ids_file:
            logger.info("Adding coding class information...")
            # Ensure no overlap between PC and lncRNA IDs
            assert (
                self.pc_ids.intersection(self.lnc_ids) == set()
            ), "PC and lncRNA ID sets overlap!"

            is_coding = self.features["transcript_id"].isin(self.pc_ids)
            is_lncrna = self.features["transcript_id"].isin(self.lnc_ids)

            # Assign values using numpy.select
            self.features["transcript_type"] = np.select(
                [is_coding, is_lncrna], ["coding", "lncRNA"], default="other"
            )
            logger.info("Coding class information added.")
            logger.info(
                f"Coding class distribution:\n{self.features['transcript_type'].value_counts()}"
            )

        logger.info(
            f"Extracted {len(self.features.columns)} features for {len(self.features)} transcripts"
        )

        return self.features

    def save_features(self):
        """Save features and summary."""
        output_file = f"{self.output_prefix}_te_features.csv"
        logger.info(f"Saving features to {output_file}")
        self.features.to_csv(output_file, index=False)
        logger.info(f"Saved {len(self.features)} transcripts to {output_file}")

        summary_file = f"{self.output_prefix}_te_summary.txt"
        logger.info(f"Saving summary to {summary_file}")
        with open(summary_file, "w") as f:
            f.write("TE Feature Extraction Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total transcripts: {len(self.features)}\n")
            f.write(f"Transcripts with TEs: {self.features['te_present'].sum()}\n")
            f.write(
                f"Percentage with TEs: {self.features['te_present'].mean() * 100:.2f}%\n\n"
            )

            if "transcript_type" in self.features.columns:
                f.write("Biotype distribution:\n")
                biotype_counts = self.features["transcript_type"].value_counts()
                for biotype, count in biotype_counts.items():
                    f.write(f"  {biotype}: {count}\n")
                f.write("\n")

            if "coding_class" in self.features.columns:
                f.write("Biotype distribution:\n")
                biotype_counts = self.features["coding_class"].value_counts()
                for biotype, count in biotype_counts.items():
                    f.write(f"  {biotype}: {count}\n")
                f.write("\n")

            f.write("TE Class distribution:\n")
            for col in ["line_count", "sine_count", "ltr_count", "dna_count"]:
                if col in self.features.columns:
                    total = self.features[col].sum()
                    f.write(f"  {col.replace('_count', '').upper()}: {int(total)}\n")
            f.write("\n")

            f.write("Feature statistics:\n")
            f.write(self.features.describe().to_string())

        logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract TE features with biotype tracking (optimized version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From RepeatMasker .out file
  %(prog)s --repeatmasker data.out --output-prefix out

  # From RepeatMasker GFF3 file with transcript BED
  %(prog)s --repeatmasker data.gff --transcripts tx.bed --output-prefix out

  # With biotype information
  %(prog)s --repeatmasker data.out --biotypes bio.tsv --output-prefix out
        """,
    )
    parser.add_argument(
        "--repeatmasker",
        required=True,
        help="RepeatMasker output file (.out or .gff/.gff3)",
    )
    parser.add_argument(
        "--transcripts",
        required=False,
        default=None,
        help="Transcript coordinates BED file (optional for .out format)",
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
    parser.add_argument("--output-prefix", required=True, help="Output file prefix")
    parser.add_argument(
        "--format",
        choices=["gff", "out", "auto"],
        default="auto",
        help="Input format (default: auto-detect)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TE Feature Extractor - Starting (Optimized Version)")
    logger.info("=" * 80)

    extractor = TEFeatureExtractor_Flexible(
        repeatmasker_file=args.repeatmasker,
        transcripts_bed=args.transcripts,
        biotypes_file=args.biotypes,
        output_prefix=args.output_prefix,
        input_format=args.format,
        pc_ids_file=args.pc_ids,
        lnc_ids_file=args.lnc_ids,
    )

    extractor.load_data()
    extractor.calculate_overlaps()
    extractor.extract_basic_features()
    extractor.save_features()

    logger.info("=" * 80)
    logger.info("Feature extraction complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
