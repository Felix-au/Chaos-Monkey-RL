.PHONY: setup test quick data train eval report ui all clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

data:
	$(PYTHON) -c "from chaosrl.data import save_calibration_report; save_calibration_report()"

train:
	$(PYTHON) -m chaosrl.train --seeds 42 43 44 45 46 --services 20 --timesteps 5000

eval:
	$(PYTHON) -m chaosrl.evaluate

quick:
	$(PYTHON) -m chaosrl.evaluate --quick
	$(PYTHON) -c "from chaosrl.analysis.report_generator import generate_report; generate_report()"

report:
	$(PYTHON) -c "from chaosrl.analysis.report_generator import generate_report; generate_report()"

all: data train eval report

ui:
	$(PYTHON) -m chaosrl.ui.server

clean:
	rm -rf .pytest_cache results/checkpoints/* results/logs/*
