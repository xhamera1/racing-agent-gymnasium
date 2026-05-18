# `notebooks/` — exploratory and reporting notebooks

| Notebook                              | Phase   | Purpose                                                                                                  |
| ------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `01_environment_exploration.ipynb`    | 1 ✅     | Inspect `CarRacing-v3`: specs, stacked-frame visualization, random-policy histogram, timings.           |
| `02_training_analysis.ipynb`          | 4 – 5   | Load `Monitor` CSVs from `experiments/`, ad-hoc plots, sanity checks.                                    |
| `03_final_report.ipynb`               | 7       | **★ Project delivery ★** — single source of truth for the PDF report.                                    |

Launch the kernel from the project `.venv` (Python **3.12**). Notebook `01`
is ready (Phase 1); `02` and `03` are added in later phases (`PLAN.md`).

> Render the final report to PDF with:
>
> ```bash
> make report
> # or:
> jupyter nbconvert --to pdf --execute notebooks/03_final_report.ipynb \
>     --output-dir reports --output final_report
> ```
