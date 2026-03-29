# TE Pipeline

Snakemake pipeline that runs RepeatMasker on GENCODE transcripts, extracts TE
features per transcript, and compares distributions between protein-coding and
lncRNA classes via statistical tests and plots.

## Requirements

- Conda / Mamba (environment: `workflow/envs/te_analysis.yaml`)
- RepeatMasker (installed in the conda env)
- Snakemake ≥ 9

## Quick start

```bash
# copy and edit the default config
cp config/config_v47.yaml config/config_mine.yaml
# edit paths: gencode_gtf, gencode_fasta, pc/lncrna FASTAs or ID files

snakemake --snakefile Snakefile \
          --configfile config/config_mine.yaml \
          --profile profiles/default \
          --use-conda

# or simply
snakemake
```

## Classification modes

Set `classification_mode` in the config to one of:

| Mode | Required keys |
|------|--------------|
| `fasta` | `pc_transcripts_fasta`, `lncrna_transcripts_fasta` |
| `id_file` | `pc_transcript_ids_file`, `lncrna_transcript_ids_file` |

## Pipeline steps

| Step | Rule | Output |
|------|------|--------|
| 1 | `parse_gencode_gtf` | `annotation/transcripts_from_gtf.bed`, `transcript_biotypes.txt` |
| 1.5 | `extract_transcript_lengths` | `annotation/transcript_lengths.txt` |
| 2 | `extract_ids_from_fasta` | `annotation/pc_transcript_ids.txt`, `lncrna_transcript_ids.txt` |
| 4 | `index_transcripts` | `annotation/all_transcripts.fa.fai` |
| 5 | `check_fasta_headers` | `annotation/all_transcripts_headers_checked.fa` |
| 5 | `run_repeatmasker_full` | `repeatmasker/all_transcripts.out{,.gff}` |
| 6 | `extract_all_features` | `features/all_transcripts_te_features.csv` |
| 7 | `filter_features_by_class` | `features/{pc,lncrna}_transcripts_te_features.csv` |
| 8 | `combine_classified_features` | `combined/classified_te_features.csv` |
| 9 | `statistical_analysis` | `analysis/univariate_tests.csv`, `summary_report.txt`, … |
| 10 | `generate_visualizations` | `plots/te_presence_comparison.png`, `volcano_plot.png`, `pca_plot.png` |

All outputs are namespaced under `results/{dataset}/`.

## Configs

| File | Purpose |
|------|---------|
| `config/config_47.yaml` | GENCODE v47 run |
| `config/config_49.yaml` | GENCODE v49 run |

## Dev info

See `te_pipeline_manifest.yaml` for the full rule catalogue, script inventory,
dataset run status, and cleanup candidates.
