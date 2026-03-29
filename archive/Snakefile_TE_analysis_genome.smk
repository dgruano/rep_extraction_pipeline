# Snakemake workflow for comprehensive TE feature extraction and analysis
# Author: Generated for bioinformatics pipeline
# Date: 2025-11-06

import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

configfile: "config.yaml"

# ============================================================================
# Global variables
# ============================================================================

SAMPLES = config.get("samples", [])
OUTPUT_DIR = config.get("output_dir", "results/te_analysis")
THREADS = config.get("threads", 4)

# Input files
GENOME_FASTA = config["genome_fasta"]
ANNOTATION_GTF = config["annotation_gtf"]
TRANSCRIPTS_FASTA = config.get("transcripts_fasta", "")

# RepeatMasker parameters
RM_SPECIES = config.get("repeatmasker_species", "human")
RM_LIB = config.get("repeatmasker_lib", "dfam")  # dfam or repbase

# ============================================================================
# Target rules
# ============================================================================

rule all:
    input:
        # Annotation outputs
        expand("{output_dir}/repeatmasker/{sample}.out.gff",
               output_dir=OUTPUT_DIR, sample=SAMPLES),

        # Feature extraction outputs
        expand("{output_dir}/features/{sample}_te_features.csv",
               output_dir=OUTPUT_DIR, sample=SAMPLES),
        expand("{output_dir}/features/{sample}_regional_te_features.csv",
               output_dir=OUTPUT_DIR, sample=SAMPLES),

        # Combined features
        f"{OUTPUT_DIR}/combined/all_samples_te_features.csv",

        # Statistical analysis
        f"{OUTPUT_DIR}/analysis/univariate_tests.csv",
        f"{OUTPUT_DIR}/analysis/pca_scores.csv",
        f"{OUTPUT_DIR}/analysis/feature_importance.csv",
        f"{OUTPUT_DIR}/analysis/summary_report.txt",

        # Visualizations
        f"{OUTPUT_DIR}/plots/te_presence_comparison.png",
        f"{OUTPUT_DIR}/plots/volcano_plot.png",
        f"{OUTPUT_DIR}/plots/pca_plot.png",

# ============================================================================
# Step 1: Prepare transcript coordinates
# ============================================================================

rule prepare_transcript_bed:
    """Extract transcript coordinates from GTF to BED format."""
    input:
        gtf = ANNOTATION_GTF
    output:
        bed = f"{OUTPUT_DIR}/annotation/transcripts.bed"
    shell:
        """
        awk 'BEGIN {{OFS="\t"}} $3=="transcript" {{
            split($10, a, "\"");
            split($14, b, "\"");
            print $1, $4-1, $5, a[2], 0, $7, b[2]
        }}' {input.gtf} > {output.bed}
        """

# ============================================================================
# Step 2: Run RepeatMasker annotation
# ============================================================================

rule run_repeatmasker:
    """Run RepeatMasker on genome or transcripts."""
    input:
        fasta = GENOME_FASTA if not TRANSCRIPTS_FASTA else TRANSCRIPTS_FASTA
    output:
        gff = f"{OUTPUT_DIR}/repeatmasker/{{sample}}.out.gff",
        out = f"{OUTPUT_DIR}/repeatmasker/{{sample}}.out",
        masked = f"{OUTPUT_DIR}/repeatmasker/{{sample}}.masked"
    params:
        species = RM_SPECIES,
        outdir = f"{OUTPUT_DIR}/repeatmasker",
        prefix = "{sample}"
    threads: THREADS
    log:
        f"{OUTPUT_DIR}/logs/repeatmasker_{{sample}}.log"
    shell:
        """
        module load repeatmasker
        RepeatMasker \
            -species {params.species} \
            -pa {threads} \
            -gff \
            -dir {params.outdir} \
            -s \
            {input.fasta} \
            2>&1 | tee {log}

        # Rename output to sample name
        mv {params.outdir}/$(basename {input.fasta}).out.gff {output.gff}
        mv {params.outdir}/$(basename {input.fasta}).out {output.out}
        mv {params.outdir}/$(basename {input.fasta}).masked {output.masked}
        """

# Alternative: Use pre-computed RepeatMasker if available
rule download_repeatmasker_annotation:
    """Download pre-computed RepeatMasker annotation from UCSC."""
    output:
        gff = f"{OUTPUT_DIR}/repeatmasker/ucsc_rmsk.gff.gz"
    params:
        ucsc_url = config.get("ucsc_repeatmasker_url",
                             "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz")
    shell:
        """
        wget -O {output.gff} {params.ucsc_url}
        """

rule convert_ucsc_rmsk_to_gff:
    """Convert UCSC RepeatMasker format to GFF3."""
    input:
        rmsk = f"{OUTPUT_DIR}/repeatmasker/ucsc_rmsk.gff.gz"
    output:
        gff = f"{OUTPUT_DIR}/repeatmasker/genome.out.gff"
    shell:
        """
        zcat {input.rmsk} | awk 'BEGIN {{OFS="\t"}} NR>1 {{
            print $6, "RepeatMasker", "repeat", $7, $8, $2, $10, ".",
                  "Target="$11";Class="$12";Family="$13";divergence="$3
        }}' > {output.gff}
        """

# ============================================================================
# Step 3: Extract TE features
# ============================================================================

rule extract_te_features:
    """Extract comprehensive TE features using custom Python script."""
    input:
        repeatmasker = f"{OUTPUT_DIR}/repeatmasker/{{sample}}.out.gff",
        transcripts = f"{OUTPUT_DIR}/annotation/transcripts.bed",
        annotation = ANNOTATION_GTF
    output:
        features = f"{OUTPUT_DIR}/features/{{sample}}_te_features.csv",
        summary = f"{OUTPUT_DIR}/features/{{sample}}_te_summary.txt"
    log:
        f"{OUTPUT_DIR}/logs/te_features_{{sample}}.log"
    shell:
        """
        python scripts/te_feature_extractor.py \
            --repeatmasker {input.repeatmasker} \
            --transcripts {input.transcripts} \
            --annotation {input.annotation} \
            --output-prefix {OUTPUT_DIR}/features/{wildcards.sample} \
            2>&1 | tee {log}
        """

rule extract_regional_te_features:
    """Extract regional TE features (UTR, CDS, introns, exons)."""
    input:
        repeatmasker = f"{OUTPUT_DIR}/repeatmasker/{{sample}}.out.gff",
        annotation = ANNOTATION_GTF
    output:
        features = f"{OUTPUT_DIR}/features/{{sample}}_regional_te_features.csv"
    log:
        f"{OUTPUT_DIR}/logs/regional_te_features_{{sample}}.log"
    shell:
        """
        python scripts/regional_te_extractor.py \
            --repeatmasker {input.repeatmasker} \
            --annotation {input.annotation} \
            --output-prefix {OUTPUT_DIR}/features/{wildcards.sample} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 4: Combine features from all samples
# ============================================================================

rule combine_te_features:
    """Combine TE features from all samples."""
    input:
        basic = expand("{output_dir}/features/{sample}_te_features.csv",
                      output_dir=OUTPUT_DIR, sample=SAMPLES),
        regional = expand("{output_dir}/features/{sample}_regional_te_features.csv",
                         output_dir=OUTPUT_DIR, sample=SAMPLES)
    output:
        combined = f"{OUTPUT_DIR}/combined/all_samples_te_features.csv"
    run:
        import pandas as pd

        # Load and combine basic features
        dfs_basic = []
        for f in input.basic:
            df = pd.read_csv(f)
            df['sample'] = Path(f).stem.replace('_te_features', '')
            dfs_basic.append(df)

        combined_basic = pd.concat(dfs_basic, ignore_index=True)

        # Load and combine regional features
        dfs_regional = []
        for f in input.regional:
            df = pd.read_csv(f)
            df['sample'] = Path(f).stem.replace('_regional_te_features', '')
            dfs_regional.append(df)

        if dfs_regional:
            combined_regional = pd.concat(dfs_regional, ignore_index=True)

            # Merge
            combined = combined_basic.merge(
                combined_regional,
                on=['transcript_id', 'sample'],
                how='left'
            )
        else:
            combined = combined_basic

        # Save
        combined.to_csv(output.combined, index=False)

# ============================================================================
# Step 5: Statistical analysis
# ============================================================================

rule statistical_analysis:
    """Perform comprehensive statistical analysis."""
    input:
        features = f"{OUTPUT_DIR}/combined/all_samples_te_features.csv"
    output:
        univariate = f"{OUTPUT_DIR}/analysis/univariate_tests.csv",
        categorical = f"{OUTPUT_DIR}/analysis/categorical_tests.csv",
        pca_scores = f"{OUTPUT_DIR}/analysis/pca_scores.csv",
        pca_loadings = f"{OUTPUT_DIR}/analysis/pca_loadings.csv",
        importance = f"{OUTPUT_DIR}/analysis/feature_importance.csv",
        summary = f"{OUTPUT_DIR}/analysis/summary_report.txt"
    log:
        f"{OUTPUT_DIR}/logs/statistical_analysis.log"
    shell:
        """
        python scripts/te_statistical_analyzer.py \
            --features {input.features} \
            --output-prefix {OUTPUT_DIR}/analysis \
            --group-column transcript_type \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 6: Generate visualizations
# ============================================================================

rule generate_visualizations:
    """Generate comprehensive visualizations."""
    input:
        features = f"{OUTPUT_DIR}/combined/all_samples_te_features.csv",
        tests = f"{OUTPUT_DIR}/analysis/univariate_tests.csv",
        pca = f"{OUTPUT_DIR}/analysis/pca_scores.csv"
    output:
        presence = f"{OUTPUT_DIR}/plots/te_presence_comparison.png",
        coverage = f"{OUTPUT_DIR}/plots/te_coverage_comparison.png",
        family = f"{OUTPUT_DIR}/plots/te_family_composition.png",
        volcano = f"{OUTPUT_DIR}/plots/volcano_plot.png",
        pca = f"{OUTPUT_DIR}/plots/pca_plot.png"
    log:
        f"{OUTPUT_DIR}/logs/visualization.log"
    shell:
        """
        python scripts/te_visualizer.py \
            --features {input.features} \
            --test-results {input.tests} \
            --pca-scores {input.pca} \
            --output-dir {OUTPUT_DIR}/plots \
            2>&1 | tee {log}
        """

# ============================================================================
# Additional utility rules
# ============================================================================

rule bedtools_intersect_validation:
    """Validate TE overlaps using bedtools (quality control)."""
    input:
        transcripts = f"{OUTPUT_DIR}/annotation/transcripts.bed",
        repeatmasker = f"{OUTPUT_DIR}/repeatmasker/{{sample}}.out.gff"
    output:
        overlaps = f"{OUTPUT_DIR}/validation/{{sample}}_te_transcript_overlaps.bed"
    shell:
        """
        bedtools intersect \
            -a {input.transcripts} \
            -b {input.repeatmasker} \
            -wao > {output.overlaps}
        """

rule generate_report:
    """Generate comprehensive HTML report."""
    input:
        features = f"{OUTPUT_DIR}/combined/all_samples_te_features.csv",
        summary = f"{OUTPUT_DIR}/analysis/summary_report.txt",
        plots = [
            f"{OUTPUT_DIR}/plots/te_presence_comparison.png",
            f"{OUTPUT_DIR}/plots/volcano_plot.png",
            f"{OUTPUT_DIR}/plots/pca_plot.png"
        ]
    output:
        report = f"{OUTPUT_DIR}/report/te_analysis_report.html"
    script:
        "scripts/generate_html_report.py"

# ============================================================================
# Clean-up rules
# ============================================================================

rule clean:
    """Remove all output files."""
    shell:
        """
        rm -rf {OUTPUT_DIR}/repeatmasker
        rm -rf {OUTPUT_DIR}/features
        rm -rf {OUTPUT_DIR}/combined
        rm -rf {OUTPUT_DIR}/analysis
        rm -rf {OUTPUT_DIR}/plots
        """

rule clean_all:
    """Remove all output including logs."""
    shell:
        """
        rm -rf {OUTPUT_DIR}
        """
