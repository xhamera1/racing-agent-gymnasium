# `reports/` — final deliverables for the lecturer

| File / folder                            | Source                                                 | When        |
| ---------------------------------------- | ------------------------------------------------------ | ----------- |
| `final_report.pdf`                       | `jupyter nbconvert notebooks/03_final_report.ipynb`    | Phase 7     |
| `figures/learning_curve_<hp>.png`        | `scripts/plot_curves.py`                               | Phase 4     |
| `figures/learning_curve_compare.png`     | `scripts/plot_curves.py`                               | Phase 4     |
| `figures/arch_A.png`, `arch_B.png`       | `utils.plotting.plot_arch_diagram`                     | Phase 5     |
| `figures/arch_compare.png`               | `scripts/plot_curves.py --include-arch`                | Phase 5     |
| `figures/timing_table.csv`               | `scripts/plot_curves.py`                               | Phase 4     |
| `figures/eval_summary.json`              | `scripts/evaluate.py --deterministic`                  | Phase 6     |
| `figures/agent_demo.mp4`                 | `scripts/record_video.py`                              | Phase 6     |

The PDF is what is submitted on MS Teams; everything in `figures/` is
regenerable from the contents of `experiments/`, so the report is
**reproducible end-to-end**.
