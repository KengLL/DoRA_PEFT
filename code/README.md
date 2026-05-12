# Code

Runnable artifacts for the project.

## Notebooks

The notebooks were authored to run on **Google Colab** with a Google Drive mount at `/content/gdrive`. They expect (and create) a working folder under `MyDrive/DoRAFinalProject/` (and `MyDrive/DoRA_Final_Project/` for the manual training notebook) where intermediate logs and `Results/` outputs are written.

Suggested execution order:

1. [`DoRA_MANUAL_v4.ipynb`](DoRA_MANUAL_v4.ipynb) — full training + evaluation pipeline. Implements the DoRA forward pass described in §3.2 of the report, runs LoRA and DoRA fine-tuning on a 20K slice of Commonsense170K across ranks 2 / 4 / 8 with 3 seeded runs, and evaluates on the 8 commonsense reasoning benchmarks (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA). Writes raw results JSON, summary CSVs, and per-row magnitude/direction arrays.

2. [`DoRA_Graph_Export.ipynb`](DoRA_Graph_Export.ipynb) — consumes the artifacts written by the training notebook (`dora_experiment_results.json`, `final_summary_table.csv`, `dora_md_per_row.npz`) and renders the figures used in the report (`bench_accuracies.png`, `md_rank_comparison.png`).

## Path / environment notes

- The notebooks reference Drive paths (e.g. `/content/gdrive/MyDrive/DoRAFinalProject/Results/`). When running outside Colab, either replicate that directory layout or edit the path constants at the top of each notebook.

- Access to the LLaMA-3.2-1B checkpoint on Hugging Face requires an authenticated token. (Can be requested on Hugging Face.)

## Reproducibility caveats

- The notebooks use 3 seeded runs per `(method, rank)` configuration as described in the report. The exact seed values live in the notebook cells: `13`, `42`, and `67`.