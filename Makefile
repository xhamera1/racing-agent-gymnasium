# Racing-Agent — convenience targets.
# Tested on Windows PowerShell (use `make.exe` from chocolatey or run commands manually).
# All Python entry points live under `scripts/`.

PYTHON ?= python
TIMESTEPS ?= 50000
SEEDS ?= 0 1 2 3 4 5 6 7 8 9

.PHONY: help setup test smoke train experiment evaluate plot video report clean

help:
	@echo "Targets:"
	@echo "  setup       - install package in editable mode"
	@echo "  test        - run pytest"
	@echo "  smoke       - 5 000-step training run to verify the pipeline"
	@echo "  train       - single training run (HP=hp_baseline, ARCH=arch_nature_cnn, SEED=0)"
	@echo "  experiment  - full HP sweep: 3 configs x 10 seeds x \$$(TIMESTEPS) steps"
	@echo "  evaluate    - deterministic evaluation of the best agent"
	@echo "  plot        - regenerate learning-curve figures"
	@echo "  video       - record a demo mp4 of the best agent"
	@echo "  report      - render notebooks/03_final_report.ipynb to PDF"
	@echo "  clean       - remove experiments/, logs/, __pycache__"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) scripts/train_single.py \
	  --hp configs/hp_baseline.yaml \
	  --arch configs/arch_nature_cnn.yaml \
	  --seed 0 --timesteps 5000

train:
	$(PYTHON) scripts/train_single.py \
	  --hp configs/$(or $(HP),hp_baseline).yaml \
	  --arch configs/$(or $(ARCH),arch_nature_cnn).yaml \
	  --seed $(or $(SEED),0) \
	  --timesteps $(TIMESTEPS)

experiment:
	$(PYTHON) scripts/run_experiment.py \
	  --configs hp_baseline hp_high_lr hp_large_batch \
	  --arch arch_nature_cnn \
	  --seeds $(SEEDS) \
	  --timesteps $(TIMESTEPS)

evaluate:
	$(PYTHON) scripts/evaluate.py --episodes 50 --deterministic

plot:
	$(PYTHON) scripts/plot_curves.py

video:
	$(PYTHON) scripts/record_video.py --episodes 3 --deterministic

report:
	jupyter nbconvert --to pdf --execute notebooks/03_final_report.ipynb \
	  --output-dir reports --output final_report

clean:
	rm -rf experiments/ logs/ **/__pycache__ .pytest_cache
