#!/bin/bash
# Setup script for TE analysis pipeline
# Creates conda environment and directory structure

set -e

echo "=========================================="
echo "TE Analysis Pipeline Setup"
echo "=========================================="

# Create conda environment
echo ""
echo "Creating conda environment..."
#conda create -n te_analysis python=3.9 -y
conda create -n te_analysis python=3.12 -y

# Activate environment
conda init bash
conda activate te_analysis

# Install Python packages
echo ""
echo "Installing Python packages..."
#pip install pandas numpy scipy scikit-learn matplotlib seaborn snakemake
conda install -c conda-forge pandas numpy scipy scikit-learn matplotlib seaborn -y
conda install -c bioconda snakemake -y

# Install bioinformatics tools via conda
echo ""
echo "Installing bioinformatics tools..."
conda install -c bioconda bedtools samtools -y

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/genome
mkdir -p data/annotation
mkdir -p scripts
mkdir -p results
mkdir -p logs

# Move Python scripts to scripts directory
echo ""
echo "Moving scripts to scripts/ directory..."
mv te_feature_extractor.py scripts/ 2>/dev/null || true
mv regional_te_extractor.py scripts/ 2>/dev/null || true
mv te_statistical_analyzer.py scripts/ 2>/dev/null || true
mv te_visualizer.py scripts/ 2>/dev/null || true

# Make scripts executable
chmod +x scripts/*.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate te_analysis"
echo "2. Install RepeatMasker manually from http://www.repeatmasker.org/"
echo "3. Download input data (genome FASTA and GTF annotation)"
echo "4. Edit config.yaml with your file paths"
echo "5. Run pipeline: snakemake -s Snakefile_TE_analysis --cores 8"
echo ""
echo "Note: RepeatMasker must be installed separately and added to PATH"
echo "      See README_TE_Pipeline.md for detailed instructions"
