# Flexible Snakemake workflow for TE feature extraction
# Supports: GTF annotation + flexible classification (FASTA or ID file)
# MODIFIED: Full GTF support, flexible classification, biotype tracking
# Author: Generated for bioinformatics pipeline
# Date: 2025-11-06

import os
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================
configfile: "config/config_v47.yaml"


# ============================================================================
# Wildcards
# ============================================================================
wildcard_constraints:
    dataset="[a-zA-Z0-9._-]+"


# ============================================================================
# Global variables
# ============================================================================
THREADS = config.get("threads", 4)

# Input files
GENCODE_GTF = config["gencode_gtf"]  # GENCODE v47 annotation - REQUIRED
GENCODE_FASTA = config.get("gencode_fasta", "")  # Full transcript FASTA (optional)

# Classification approach
CLASSIFICATION_MODE = config.get("classification_mode", "fasta")  # "fasta" or "id_file"

# Conditional inputs based on classification mode
if CLASSIFICATION_MODE == "fasta":
    PC_TRANSCRIPTS_FA = config.get("pc_transcripts_fasta", "")
    LNCRNA_TRANSCRIPTS_FA = config.get("lncrna_transcripts_fasta", "")
elif CLASSIFICATION_MODE == "id_file":
    PC_TRANSCRIPT_IDS = config.get("pc_transcript_ids_file", "")
    LNCRNA_TRANSCRIPT_IDS = config.get("lncrna_transcript_ids_file", "")

# RepeatMasker parameters
RM_SPECIES = config.get("repeatmasker_species", "human")

# ============================================================================
# Validation and setup
# ============================================================================
if not os.path.exists(GENCODE_GTF):
    raise ValueError(f"GENCODE GTF not found: {GENCODE_GTF}")

if CLASSIFICATION_MODE == "fasta":
    if not PC_TRANSCRIPTS_FA or not LNCRNA_TRANSCRIPTS_FA:
        raise ValueError(
            "fasta mode requires both pc_transcripts_fasta and lncrna_transcripts_fasta"
        )
    if not os.path.exists(PC_TRANSCRIPTS_FA) or not os.path.exists(LNCRNA_TRANSCRIPTS_FA):
        raise ValueError("Transcript FASTA files not found")

elif CLASSIFICATION_MODE == "id_file":
    if not PC_TRANSCRIPT_IDS or not LNCRNA_TRANSCRIPT_IDS:
        raise ValueError(
            "id_file mode requires both pc_transcript_ids_file and lncrna_transcript_ids_file"
        )
    if not os.path.exists(PC_TRANSCRIPT_IDS) or not os.path.exists(LNCRNA_TRANSCRIPT_IDS):
        raise ValueError("Transcript ID files not found")


# ============================================================================
# Target rules
# ============================================================================
# Get list of datasets from config
DATASETS = config.get("datasets", ["default"])

rule all:
    input:
        expand(
            [
                # GTF parsing
                "results/{dataset}/annotation/transcripts_from_gtf.bed",
                "results/{dataset}/annotation/transcript_biotypes.txt",
                "results/{dataset}/annotation/transcript_lengths.txt",
                # Classification (depends on mode)
                "results/{dataset}/annotation/pc_transcript_ids.txt",
                "results/{dataset}/annotation/lncrna_transcript_ids.txt",
                # RepeatMasker on full database
                "results/{dataset}/repeatmasker/all_transcripts.out.gff",
                "results/{dataset}/repeatmasker/all_transcripts.out",
                # Feature extraction on full database
                "results/{dataset}/features/all_transcripts_te_features.csv",
                "results/{dataset}/analysis/univariate_tests.csv",
                # Statistical analysis and visualization
                "results/{dataset}/analysis/summary_report.txt",
                "results/{dataset}/plots/hit_presence_comparison.png",
            ],
            dataset=DATASETS,
        )


# ============================================================================
# Step 1: Parse GENCODE GTF - Extract transcripts and biotypes
# ============================================================================
rule parse_gencode_gtf:
    """Parse GENCODE GTF to extract transcript information."""
    input:
        gtf=GENCODE_GTF,
    output:
        bed="results/{dataset}/annotation/transcripts_from_gtf.bed",
        biotypes="results/{dataset}/annotation/transcript_biotypes.txt",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/parse_gtf.log",
    resources:
        mem_mb=16000,
    shell:
        """
        python workflow/scripts/parse_gencode_gtf.py \
            --gtf {input.gtf} \
            --output-bed {output.bed} \
            --output-biotypes {output.biotypes} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 1.5: Extract transcript lengths from GTF (for later use in feature extraction)
# ============================================================================
rule extract_transcript_lengths:
    """Extract transcript lengths from GTF for feature extraction."""
    input:
        gtf=GENCODE_GTF,
    output:
        lengths="results/{dataset}/annotation/transcript_lengths.txt",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/extract_lengths.log",
    resources:
        mem_mb=16000,
    shell:
        """
        awk -F"\\t" '$3 == "exon" {{
            match($9, /transcript_id "([^"]+)"/, arr);
            L[arr[1]] += $5 - $4 + 1
        }} END {{
            for (t in L) print t "\\t" L[t]
        }}' {input.gtf} > {output.lengths}
        """


# ============================================================================
# Step 2: Handle Classification - FASTA or ID file mode
# ============================================================================
# MODE A: Extract IDs from FASTA files
rule extract_ids_from_fasta:
    """Extract transcript IDs from FASTA files (FASTA mode)."""
    input:
        pc_fa=PC_TRANSCRIPTS_FA if CLASSIFICATION_MODE == "fasta" else [],
        lncrna_fa=LNCRNA_TRANSCRIPTS_FA if CLASSIFICATION_MODE == "fasta" else [],
    output:
        pc_ids="results/{dataset}/annotation/pc_transcript_ids.txt",
        lncrna_ids="results/{dataset}/annotation/lncrna_transcript_ids.txt",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/extract_ids.log",
    run:
        if CLASSIFICATION_MODE == "fasta":
            shell(
                """
            # Extract IDs from pc_transcripts.fa
            grep "^>" {input.pc_fa} | \
                sed 's/^>//g' | \
                cut -d'|' -f1 > {output.pc_ids}

            # Extract IDs from lncrna_transcripts.fa
            grep "^>" {input.lncrna_fa} | \
                sed 's/^>//g' | \
                cut -d'|' -f1 > {output.lncrna_ids}
            """
            )
        else:
            # If using ID file mode, just copy the files
            shell("cp {PC_TRANSCRIPT_IDS} {output.pc_ids}")
            shell("cp {LNCRNA_TRANSCRIPT_IDS} {output.lncrna_ids}")


# ============================================================================
# Step 3: Index and prepare for RepeatMasker
# ============================================================================
rule index_transcripts:
    """Index transcript FASTA for RepeatMasker."""
    input:
        fa="results/{dataset}/annotation/all_transcripts.fa",
    output:
        fai="results/{dataset}/annotation/all_transcripts.fa.fai",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/index_transcripts.log",
    shell:
        """
        samtools faidx {input.fa} 2>&1 | tee {log}
        """


# ============================================================================
# Step 4: Run RepeatMasker on full database
# ============================================================================
rule check_fasta_headers:
    """Check and clean FASTA headers for RepeatMasker compatibility."""
    input:
        fa="results/{dataset}/annotation/all_transcripts.fa",
    output:
        cleaned_fa="results/{dataset}/annotation/all_transcripts_headers_checked.fa",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/check_fasta_headers.log",
    shell:
        """
        python workflow/scripts/check_fasta_headers.py \
            --input {input.fa} \
            --output {output.cleaned_fa} \
            2>&1 | tee {log}
        """


rule run_repeatmasker_full:
    """Run RepeatMasker on all transcripts."""
    input:
        fa="results/{dataset}/annotation/all_transcripts_headers_checked.fa",
        fai="results/{dataset}/annotation/all_transcripts.fa.fai",
    output:
        gff="results/{dataset}/repeatmasker/all_transcripts.out.gff",
        out="results/{dataset}/repeatmasker/all_transcripts.out",
    conda:
        "workflow/envs/te_analysis.yaml"
    params:
        species=RM_SPECIES,
        outdir=lambda wc, output: subpath(output.gff, parent=True),
    threads: THREADS
    resources:
        mem_mb=200000,
        runtime="3d",
    log:
        "logs/{dataset}/repeatmasker_full.log",
    benchmark:
        "benchmarks/{dataset}/repeatmasker_full.txt",
    shell:
        """
        RepeatMasker \
            -species {params.species} \
            -pa {threads} \
            -gff \
            -dir {params.outdir} \
            -s \
            {input.fa} \
            2>&1 | tee {log}

        mv {params.outdir}/$(basename {input.fa}).out.gff {output.gff}
        mv {params.outdir}/$(basename {input.fa}).out {output.out}
        """


# ============================================================================
# Step 5: Extract TE features for full database
# ============================================================================
rule extract_all_features:
    """Extract TE features for all transcripts."""
    input:
        repeatmasker="results/{dataset}/repeatmasker/all_transcripts.out",
        bed="results/{dataset}/annotation/transcripts_from_gtf.bed",
        biotypes="results/{dataset}/annotation/transcript_biotypes.txt",
        lengths="results/{dataset}/annotation/transcript_lengths.txt",
        pc_ids="results/{dataset}/annotation/pc_transcript_ids.txt",
        lncrna_ids="results/{dataset}/annotation/lncrna_transcript_ids.txt",
    output:
        features="results/{dataset}/features/all_transcripts_te_features.csv",
        summary="results/{dataset}/features/all_transcripts_te_summary.txt",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/extract_features_all.log",
    shell:
        """
        python workflow/scripts/te_feature_extractor.py \
            --repeatmasker {input.repeatmasker} \
            --transcripts {input.bed} \
            --biotypes {input.biotypes} \
            --lengths {input.lengths} \
            --pc-ids {input.pc_ids} \
            --lnc-ids {input.lncrna_ids} \
            --output-prefix results/{wildcards.dataset}/features/all_transcripts \
            2>&1 | tee {log}
        """


# ============================================================================
# Step 6: Filter features by classification
# ============================================================================
rule filter_features_by_class:
    """Filter TE features for pc vs lncRNA classes."""
    input:
        features="results/{dataset}/features/all_transcripts_te_features.csv",
        pc_ids="results/{dataset}/annotation/pc_transcript_ids.txt",
        lncrna_ids="results/{dataset}/annotation/lncrna_transcript_ids.txt",
    output:
        pc_features="results/{dataset}/features/pc_transcripts_te_features.csv",
        lncrna_features="results/{dataset}/features/lncrna_transcripts_te_features.csv",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/filter_features.log",
    shell:
        """
        python workflow/scripts/filter_features_by_class.py \
            --features {input.features} \
            --pc-ids {input.pc_ids} \
            --lncrna-ids {input.lncrna_ids} \
            --output-pc {output.pc_features} \
            --output-lncrna {output.lncrna_features} \
            2>&1 | tee {log}
        """


# ============================================================================
# Step 7: Combine features and prepare for analysis
# ============================================================================
rule combine_classified_features:
    """Combine pc and lncRNA features with group labels."""
    input:
        pc_features="results/{dataset}/features/pc_transcripts_te_features.csv",
        lncrna_features="results/{dataset}/features/lncrna_transcripts_te_features.csv",
    output:
        combined="results/{dataset}/combined/classified_te_features.csv",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/combine_features.log",
    run:
        import pandas as pd

        pc_df = pd.read_csv(input.pc_features)
        lncrna_df = pd.read_csv(input.lncrna_features)

        pc_df["group"] = "Coding"
        lncrna_df["group"] = "lncRNA"

        combined = pd.concat([pc_df, lncrna_df], ignore_index=True)
        combined.to_csv(output.combined, index=False)
        print("Combined features saved to {output.combined}")


# ============================================================================
# Step 8: Statistical analysis
# ============================================================================
rule statistical_analysis:
    """Perform statistical analysis on classified features."""
    input:
        features="results/{dataset}/features/all_transcripts_te_features.csv",
    output:
        univariate="results/{dataset}/analysis/univariate_tests.csv",
        categorical="results/{dataset}/analysis/categorical_tests.csv",
        pca_scores="results/{dataset}/analysis/pca_scores.csv",
        summary="results/{dataset}/analysis/summary_report.txt",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/statistical_analysis.log",
    shell:
        """
        python workflow/scripts/te_statistical_analyzer.py \
            --features {input.features} \
            --output-prefix results/{wildcards.dataset}/analysis \
            2>&1 | tee {log}
        """


# ============================================================================
# Step 9: Visualizations
# ============================================================================
rule generate_visualizations:
    """Generate comprehensive visualizations."""
    input:
        features="results/{dataset}/features/all_transcripts_te_features.csv",
        tests="results/{dataset}/analysis/univariate_tests.csv",
        pca="results/{dataset}/analysis/pca_scores.csv",
    output:
        presence="results/{dataset}/plots/te_presence_comparison.png",
        volcano="results/{dataset}/plots/volcano_plot.png",
        pca="results/{dataset}/plots/pca_plot.png",
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/visualization.log",
    shell:
        """
        python workflow/scripts/te_visualizer.py \
            --features {input.features} \
            --test-results {input.tests} \
            --pca-scores {input.pca} \
            --output-dir results/{wildcards.dataset}/plots \
            2>&1 | tee {log}
        """


# ============================================================================
# Clean-up rules
# ============================================================================
rule clean:
    """Remove all output files for a dataset."""
    conda:
        "workflow/envs/te_analysis.yaml"
    log:
        "logs/{dataset}/clean.log",
    shell:
        """
        rm -rf results/{wildcards.dataset}/*
        echo "All output files removed for dataset {wildcards.dataset}." 2>&1 | tee {log}
        """
