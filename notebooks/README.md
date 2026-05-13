# `notebooks/` — exploratory and reporting notebooks

| Notebook                              | Phase   | Purpose                                                                                                  |
| ------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `01_environment_exploration.ipynb`    | 1       | Inspect `CarRacing-v3`: observation / action specs, render a few frames, measure avg step & episode time.|
| `02_training_analysis.ipynb`          | 4 – 5   | Load `Monitor` CSVs from `experiments/`, ad-hoc plots, sanity checks.                                    |
| `03_final_report.ipynb`               | 7       | **★ Project delivery ★** — single source of truth for the PDF report.                                    |

To create a notebook, open the corresponding file in Jupyter / VS Code /
Cursor and start a fresh kernel using `.venv (Python 3.12)`. The files are
intentionally not committed at bootstrap; they will be authored as their
parent phases land (see `PLAN.md`).

> Render the final report to PDF with:
>
> ```bash
> make report
> # or:
> jupyter nbconvert --to pdf --execute notebooks/03_final_report.ipynb \
>     --output-dir reports --output final_report
> ```
