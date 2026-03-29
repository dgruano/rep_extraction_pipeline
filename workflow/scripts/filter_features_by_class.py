#!/usr/bin/env python3
"""
Filter TE Features by Classification
=====================================
Filter full TE feature matrix by transcript class (pc vs lncRNA)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureFilterByClass:
    """Filter features by transcript class."""

    def __init__(self, features_file: str, pc_ids_file: str, lncrna_ids_file: str):
        self.features_file = features_file
        self.pc_ids_file = pc_ids_file
        self.lncrna_ids_file = lncrna_ids_file

        self.features = None
        self.pc_ids = None
        self.lncrna_ids = None

    def load_data(self):
        """Load all data files."""
        logger.info("Loading data files...")

        # Load features
        self.features = pd.read_csv(self.features_file)
        logger.info(f"Loaded {len(self.features)} transcripts with features")

        # Load PC transcript IDs
        with open(self.pc_ids_file, "r") as f:
            self.pc_ids = set(line.strip() for line in f)
        logger.info(f"Loaded {len(self.pc_ids)} protein-coding transcript IDs")

        # Load lncRNA transcript IDs
        with open(self.lncrna_ids_file, "r") as f:
            self.lncrna_ids = set(line.strip() for line in f)
        logger.info(f"Loaded {len(self.lncrna_ids)} lncRNA transcript IDs")

    def filter_features(self):
        """Filter features by class."""
        logger.info("Filtering features by class...")

        # Normalize transcript IDs in feature matrix
        self.features["tx_id_base"] = (
            self.features["transcript_id"].str.split(".").str[0]
        )

        # Filter PC transcripts
        pc_features = self.features[
            self.features["tx_id_base"].isin(self.pc_ids)
        ].copy()

        # Filter lncRNA transcripts
        lncrna_features = self.features[
            self.features["tx_id_base"].isin(self.lncrna_ids)
        ].copy()

        logger.info(f"PC transcripts: {len(pc_features)}")
        logger.info(f"lncRNA transcripts: {len(lncrna_features)}")

        # Check for transcripts in neither class
        unclassified = len(self.features) - len(pc_features) - len(lncrna_features)
        if unclassified > 0:
            logger.warning(f"Unclassified transcripts: {unclassified}")

        return pc_features, lncrna_features

    def save_filtered(
        self,
        pc_features: pd.DataFrame,
        lncrna_features: pd.DataFrame,
        output_pc: str,
        output_lncrna: str,
    ):
        """Save filtered feature matrices."""
        # Remove temporary column
        pc_features = pc_features.drop(columns=["tx_id_base"])
        lncrna_features = lncrna_features.drop(columns=["tx_id_base"])

        # Add group column
        pc_features["group"] = "coding"
        lncrna_features["group"] = "lncRNA"

        # Save
        combined = pd.concat([pc_features, lncrna_features], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Filter TE features by transcript class"
    )
    parser.add_argument("--features", required=True, help="Full feature matrix CSV")
    parser.add_argument("--pc-ids", required=True, help="PC transcript IDs file")
    parser.add_argument(
        "--lncrna-ids", required=True, help="lncRNA transcript IDs file"
    )
    parser.add_argument("--output-pc", required=True, help="Output PC features CSV")
    parser.add_argument(
        "--output-lncrna", required=True, help="Output lncRNA features CSV"
    )

    args = parser.parse_args()

    filter_obj = FeatureFilterByClass(args.features, args.pc_ids, args.lncrna_ids)
    filter_obj.load_data()
    pc_feat, lncrna_feat = filter_obj.filter_features()
    filter_obj.save_filtered(pc_feat, lncrna_feat, args.output_pc, args.output_lncrna)

    logger.info("Feature filtering complete!")


if __name__ == "__main__":
    main()
