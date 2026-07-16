# Beyond AUC -- cost-sensitive, calibrated, drift-robust CNP fraud detection
# All stages are idempotent; later stages read artifacts persisted by earlier
# ones. See README section 7 for the expected end-to-end runtime.

PY := python3

.PHONY: all data features train evaluate figures notebooks app app-artifacts test clean

all: data features train evaluate figures test

data:
	$(PY) -m src.data.prepare

features:
	$(PY) -m src.features.build
	$(PY) -m src.data.eda

train:
	$(PY) -m src.models.train

evaluate:
	$(PY) -m src.evaluation.calibration
	$(PY) -m src.evaluation.cost
	$(PY) -m src.evaluation.ablation
	$(PY) -m src.evaluation.drift
	$(PY) -m src.evaluation.final_test

figures:
	$(PY) -m src.visualization.figures

notebooks:
	$(PY) -m src.visualization.make_notebooks --execute

pdfs:
	$(PY) -m src.visualization.make_pdfs README.md reports/pdf/Beyond_AUC_paper.pdf
	$(PY) -m src.visualization.make_pdfs RUN_REPORT.md reports/pdf/Run_report.pdf
	$(PY) -m src.visualization.make_pdfs reports/companions/companion_paper.md reports/pdf/Beyond_AUC_plain_english_companion.pdf
	$(PY) -m src.visualization.make_pdfs reports/companions/companion_run_report.md reports/pdf/Run_report_plain_english_companion.pdf

app:
	streamlit run app/app.py

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf data/interim/* data/processed/*
