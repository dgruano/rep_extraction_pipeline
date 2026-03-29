#!/usr/bin/env python3
"""
GENCODE GTF Parser
==================
Extract transcript IDs, biotypes, and coordinates from GENCODE GTF.
Supports: Full GTF parsing with comprehensive biotype tracking
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GENCODEGTFParser:
    """Parse GENCODE GTF files with comprehensive transcript information."""

    def __init__(self, gtf_file: str):
        self.gtf_file = gtf_file
        self.gtf = None
        self.transcripts = None
        self.biotypes = None

    def load_gtf(self):
        """Load GTF file."""
        logger.info(f"Loading GTF: {self.gtf_file}")

        gtf_cols = [
            "chrom",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attributes",
        ]

        self.gtf = pd.read_csv(
            self.gtf_file,
            sep="\t",
            comment="#",
            header=None,
            names=gtf_cols,
            dtype={"start": int, "end": int},
        )

        logger.info(f"Loaded {len(self.gtf)} GTF features")

    def parse_attributes(self):
        """Parse GTF attributes column."""
        logger.info("Parsing GTF attributes...")

        # Extract key attributes from 9th column
        self.gtf["transcript_id"] = self.gtf["attributes"].str.extract(
            r'transcript_id "([^"]+)"'
        )
        self.gtf["transcript_id_base"] = self.gtf["transcript_id"].str.split(".").str[0]

        self.gtf["gene_id"] = self.gtf["attributes"].str.extract(r'gene_id "([^"]+)"')

        self.gtf["gene_type"] = self.gtf["attributes"].str.extract(
            r'gene_type "([^"]+)"'
        )

        self.gtf["transcript_type"] = self.gtf["attributes"].str.extract(
            r'transcript_type "([^"]+)"'
        )

        self.gtf["transcript_name"] = self.gtf["attributes"].str.extract(
            r'transcript_name "([^"]+)"'
        )

    def extract_transcripts(self):
        """Extract transcript-level information."""
        logger.info("Extracting transcript information...")

        # Get transcript features
        transcript_features = self.gtf[self.gtf["feature"] == "transcript"].copy()

        # Create BED output
        # Create BED output
        self.transcripts = transcript_features[
            [
                "chrom",
                "start",
                "end",
                "transcript_id_base",
                "score",
                "strand",
                "transcript_id",
                "transcript_type",
                "gene_id",
                "transcript_name",
                "gene_type",
            ]
        ].drop_duplicates(subset=["transcript_id"])
        # Adjust coordinates for BED (0-based)
        self.transcripts["start"] = self.transcripts["start"] - 1

        logger.info(f"Extracted {len(self.transcripts)} unique transcripts")

        return self.transcripts

    def extract_biotypes(self):
        """Extract comprehensive biotype information."""
        logger.info("Extracting biotype information...")

        self.biotypes = self.transcripts[
            [
                "transcript_id_base",
                "transcript_id",
                "transcript_type",
                "gene_type",
                "transcript_name",
                "gene_id",
            ]
        ].drop_duplicates()

        # Count occurrences
        logger.info("\nBiotype distribution:")
        biotype_counts = self.biotypes["transcript_type"].value_counts()
        for biotype, count in biotype_counts.items():
            logger.info(f"  {biotype}: {count}")

        return self.biotypes

    def save_outputs(self, output_bed: str, output_biotypes: str):
        """Save BED and biotype files."""
        # Save BED file
        bed_output = self.transcripts[
            ["chrom", "start", "end", "transcript_id", "score", "strand"]
        ]

        bed_output.to_csv(output_bed, sep="\t", index=False, header=False)
        logger.info(f"Saved BED file: {output_bed}")

        # Save biotype file with header
        self.biotypes.to_csv(output_biotypes, sep="\t", index=False)
        logger.info(f"Saved biotype file: {output_biotypes}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse GENCODE GTF and extract transcript information"
    )
    parser.add_argument("--gtf", required=True, help="GENCODE GTF file")
    parser.add_argument("--output-bed", required=True, help="Output BED file")
    parser.add_argument("--output-biotypes", required=True, help="Output biotype file")

    args = parser.parse_args()

    gtf_parser = GENCODEGTFParser(args.gtf)
    gtf_parser.load_gtf()
    gtf_parser.parse_attributes()
    gtf_parser.extract_transcripts()
    gtf_parser.extract_biotypes()
    gtf_parser.save_outputs(args.output_bed, args.output_biotypes)

    logger.info("GTF parsing complete!")


if __name__ == "__main__":
    main()
