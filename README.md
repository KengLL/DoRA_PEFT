
# DoRA: Weight-Decomposed Low-Rank Adaptation — Reproduction

CS 5782 Final Project. A reproduction study of **DoRA: Weight-Decomposed Low-Rank Adaptation** (Liu et al., 2024) on commonsense reasoning tasks.

## Introduction

Fine-tuning large language models is expensive because it requires updating all model parameters. Parameter-efficient fine-tuning (PEFT) methods such as LoRA reduce this cost by training low-rank adapters, but LoRA couples weight magnitude and direction, limiting performance relative to full fine-tuning [2].

DoRA (Liu et al., 2024) addresses this limitation by decomposing pretrained weights into separate magnitude and direction components and tuning them independently [1]. This project reproduces DoRA’s reported improvements over LoRA on commonsense reasoning benchmarks.

## Chosen Result

We reproduced the LoRA and DoRA results on 8 commonsense reasoning tasks: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e/c, and OBQA — a subset of Table 1 from the original paper.

We also reproduced the magnitude/direction update behavior from Figure 2, validating the paper’s core claim that DoRA better decouples weight magnitude and direction than LoRA, especially at low ranks.

## GitHub Contents

```
README.md — this overview
code/ — training + figure-export notebooks (ipynb)
data/ — dataset acquisition notes
results/ — generated figures, tables, logs
poster/ — DoRA_Poster.pdf
report/ — Final_Report_draft.docx
LICENSE — MIT
.gitignore
```

The two notebooks under [`code/`](code/) are the primary entry points.

## Re-implementation Details

-   **Model.** LLaMA-3.2-1B [3] (smaller than the LLaMA-7B used in the original paper, chosen due to compute constraints).
-   **Training data.** A 20K training slice of Commonsense170K [3].
-   **Ranks evaluated.** 8, 4, 2 — lower than the original paper to accommodate for smaller model rank-sensitive accuracy differences on the smaller backbone.
-   **Hyperparameters.** Following the original paper: α = 2·rank, dropout = 0.05, AdamW optimizer with lr = 2e-4, 3 epochs.
-   **What we done differently.** batch size 16, gradient accumulation of 4 [1]. 3 seeded runs per configuration for statistical stability. (A100 GPU)
-   **Forward pass.** Input `x` is passed through dropout, then splits into three parallel paths whose outputs are summed: (i) the frozen base output `F.linear(W₀, x, b)`, pre-computed once per forward pass; (ii) a direction correction `F.linear(W₀, ·)` scaled by `(G − 1)`; and (iii) the adapter output `F.linear(sBA, ·)` scaled by `G`, where `G = m / ‖W₀ + sBA‖_row` is the per-row magnitude rescaling factor. `G` and `sBA` are computed inside `torch.no_grad()` so the row-norm denominator is detached from the computational graph, preserving the independence of magnitude and direction during training (per §4 of the original paper).

Note: The entire notebook should take around 16 hours to run on A100 GPU.

## Reproduction Steps

The pipeline lives in two notebooks under [`code/`](code/):

1.  Open [`code/DoRA_MANUAL_v4.ipynb`](code/DoRA_MANUAL_v4.ipynb) in Google Colab. The notebook mounts Google Drive at `/content/gdrive` and reads from / writes to `MyDrive/DoRA_Final_Project/`. Run it end to end to train LoRA and DoRA at ranks 2, 4, 8 (3 seeded runs each) and evaluate on the 8 commonsense reasoning benchmarks. Outputs are written to `MyDrive/DoRAFinalProject/Results/`.
2.  Open [`code/DoRA_Graph_Export.ipynb`](code/DoRA_Graph_Export.ipynb) in Colab against the same Drive. It consumes the artifacts produced by step 1 (`dora_experiment_results.json`, `final_summary_table.csv`, `dora_md_per_row.npz`) and renders `bench_accuracies.png` and `md_rank_comparison.png`.

Caveats:

-   The notebooks reference the Drive paths. To run outside Colab, edit the path constants at the top of each notebook.
-   Access to LLaMA-3.2-1B on Hugging Face requires an authenticated token. (Can be easily requested on HuggingFace)

## Results / Insights

-   The average accuracy of our DoRA implementation across the 8 commonsense reasoning tasks exceeds the LoRA baseline across all of ranks 8, 4, 2. The difference is minimal at rank 8 and more significant at rank 2.
-   Rank 16 was initially tested to match the original paper, but the performance gap was minimal — possibly due to capacity saturation of the 1B model leaving little room for DoRA's decomposition to help. Reducing the rank surfaced more visible LoRA / DoRA differences.
-   Variance across seeded runs causes overlaps between LoRA and DoRA in some per-benchmark cases, so not every observed gap is statistically significant.
-   Magnitude/direction update plots (Figure 2 in the report) show LoRA's components are entangled (strong positive correlation between magnitude and direction update size), while DoRA's trend line is nearly flat and slightly negative — consistent with the original paper and indicating successful decoupling.
-   At rank 2, LoRA drops below the unadapted baseline (44.05%) while DoRA remains above it, indicating that DoRA's magnitude-direction decomposition provides meaningful stability under severe parameter constraints.

## Conclusion

From our experiment, DoRA consistently outperforms LoRA across all ranks while only adding marginal extra parameters over LoRA — a negligible overhead. The gap widens at low rank (r = 2), where LoRA degrades more. DoRA is a drop-in replacement for LoRA with no architecture change required, making it a simple accuracy booster. We learned that higher ranks for small models could experience capacity saturation, reducing DoRA’s benefits. Fine-tuning is sensitive to initialization, especially at lower ranks. 

For future directions, we hope to continue to investigate other PEFT methods, such as magnitude-aware PEFT methods (MAP). We also hope to extend the investigation to domains beyond text — including vision and audio — to assess whether DoRA's advantages persist across modalities.

## References

1.  Liu et al. (2024). _DoRA: Weight-Decomposed Low-Rank Adaptation._ arXiv:2402.09353.
2.  Hu et al. (2022). _LoRA: Low-Rank Adaptation of Large Language Models._ ICLR 2022.
3.  Meta AI (2024). _Llama 3.2._ Hugging Face.
4.  He, Z. (2024). Commonsense170K. 

