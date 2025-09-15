## Twitter Sentiment Analysis

Modular, reproducible sentiment analysis for tweets with a clean CLI, improved preprocessing, multiple model options, artifacts saving, and unit tests. Backward‑compatible with the original single‑file script.

---

## Overview

- Preprocesses tweets with a fast regex tokenizer (lowercasing, <url>/<user> placeholders, hashtag stripping, elongated‑char normalization, simple contractions).
- Vectorizes with TF‑IDF over word n‑grams (optional char n‑grams).
- Trains configurable models: Naive Bayes, Logistic Regression, or Linear SVM.
- Splits training data deterministically with stratify and a fixed seed.
- Saves a trained pipeline, metrics (JSON), and figures (EDA + confusion matrices).
- Includes small unit tests for core components.

---

## Project Structure

- `twitter_sentiment/` — package
  - `data.py` — dataset loading, outputs directory management, EDA plots.
  - `preprocess.py` — regex‑based normalization and tokenization.
  - `model.py` — vectorizer and model builders; small default param grids.
  - `evaluate.py` — metrics computation and confusion‑matrix plotting.
  - `train.py` — CLI for training, validation, test evaluation, and saving artifacts.
  - `predict.py` — CLI for inference from text or CSV.
- `tests/` — unit tests for preprocessing, data loading, and metrics.
- `requirements.txt` — runtime dependencies.
- `sentanal.py` — legacy entry point delegating to `twitter_sentiment.train`.
- `data/train.csv`, `data/test.csv` — example datasets (see Data Schema).
- `outputs/` — generated artifacts (created on first run).

---

## Setup

- Python 3.9–3.11 recommended (works in your conda env `test_env`).
- Install dependencies: `pip install -r requirements.txt` (or conda equivalents).
- No NLTK/TextBlob corpora required (the tokenizer is self‑contained).
- Headless environments: set `MPLCONFIGDIR` to a writable folder to silence font cache warnings, e.g. `.mplconfig`.
 - Data location: put `train.csv` and `test.csv` in `data/` (preferred) or project root. The CLI auto-detects `data/train.csv` and `data/test.csv` if present.
 - Git hygiene: ephemeral outputs and caches are ignored by default via `.gitignore` (e.g., `outputs*/`, `.mplconfig/`, `__pycache__/`, `.idea/`, `nltk_data/`). Keep only source and docs under version control.
  - If such files were previously committed, untrack with e.g. `git rm -r --cached outputs .mplconfig __pycache__ .idea`.

Conda users
- Use your prepared env: `conda run -n test_env ...`
- Examples:
  - `conda run -n test_env python -m unittest -v`
  - `conda run -n test_env streamlit run app/streamlit_app.py`
  - To silence Matplotlib cache warnings: `export MPLCONFIGDIR=$(pwd)/.mplconfig`

---

## Data Schema

- Train CSV: columns `label` (0/1) and `tweet` (string) are required.
- Test CSV: requires `tweet`; `label` is optional (if present, test metrics are reported).

---

## Usage

- Train and evaluate
```
python -m twitter_sentiment.train \
  --train data/train.csv \
  --test data/test.csv \
  --outputs outputs \
  --model nb            # or: logreg, linearsvm
# optional flags:
# --use-char-ngrams         # add char n-grams (3–5)
# --word-ngrams 1,2         # set word n-gram range (default 1,2)
# --no-grid                 # disable small GridSearchCV
# --balance downsample      # class balancing: none|downsample|upsample
# --calibrate-threshold     # tune decision threshold on validation
# --test-size 0.2 --seed 42 # control split and seed
```

- Predict with a saved model
```
python -m twitter_sentiment.predict --model outputs/artifacts/model_pipeline.joblib --text "I love this!"
# or batch from CSV with a 'tweet' column
python -m twitter_sentiment.predict --model outputs/artifacts/model_pipeline.joblib --csv data/test.csv --out-csv preds.csv
```

Notes
- If your `test.csv` contains a `label` column with missing values, metrics are computed on the labeled subset; predictions are saved for all rows.
- Legacy artifacts pickled with a `src.*` module may fail to load via the raw CLI; the Streamlit app includes a compatibility shim and can load them. You can then re-save a fresh pipeline for future use.

---

## Outputs

- `outputs/artifacts/model_pipeline.joblib` — trained sklearn pipeline.
- `outputs/reports/metrics.json` — validation metrics (and test metrics if labels exist).
- `outputs/figures/` — EDA plots (`barplot_label_length.png`, `label_counts.png`) and confusion matrices (`confusion_matrix_*.png`).

---

## Testing

```
# direct
python -m unittest discover -s tests -p "test_*.py" -v

# conda
conda run -n test_env python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Troubleshooting

- Font cache warnings: set `MPLCONFIGDIR` to a writable path, e.g. `export MPLCONFIGDIR=$(pwd)/.mplconfig`.
- If `test.csv` lacks labels, only predictions are produced (no test metrics).

---

## Current Status

- Implemented
  - Correct metrics ordering (y_true, y_pred) and macro‑F1 reporting.
  - Deterministic train/val split with `stratify` and `random_state`.
  - Self‑contained, fast preprocessing (no external corpora).
  - Word n‑grams (1–2 default) and optional char n‑grams (3–5).
  - Models: NB (default), Logistic Regression, Linear SVM; small optional grid search.
  - Saved artifacts and figures; metrics in JSON; simple unit tests.

- Known limitations
  - Class imbalance handling beyond linear models: NB not reweighted; no sampling yet.
  - Hyperparameter search is intentionally small; no cross‑validated model selection across vectorizers.
  - No experiment tracker; logging is minimal; no CLI subcommands.
  - No CI pipeline or pre‑commit hooks.
  - Dataset documentation and bias considerations are minimal.

---

## Roadmap

1) Evaluation & Reporting (short term)
- Expand metrics JSON: add per‑class precision/recall/F1 explicitly and ROC‑AUC (when applicable).
- Save validation predictions and probabilities to `outputs/reports/val_predictions.csv`.
- Add option to save test predictions to `outputs/reports/test_predictions.csv`.

2) Modeling & Imbalance (short term)
- Add class weighting or calibrated thresholds for NB; optionally integrate simple resampling strategies (e.g., class‑balanced subsampling) behind a flag.
- Broaden hyperparameter grids and add cross‑validation for model selection (limited, time‑boxed).

3) Preprocessing (short term)
- Optional emoji and emoticon normalization tables; configurable placeholder policy; configurable repeat‑char limit.

4) Developer Experience (medium term)
- Introduce structured logging (Python `logging`) with `--log-level` and file output.
- Add a `config.yaml` option and CLI override strategy; persist resolved config next to artifacts.
- Provide `requirements-dev.txt` and pre‑commit hooks (`ruff`, `black`, `isort`).
- Set up GitHub Actions CI to run tests and lint on pushes/PRs.

5) Streamlit App (medium term)
- Build a Streamlit UI to:
  - Enter text and see predictions + class probabilities
  - Upload a CSV and download predictions
  - Visualize EDA and confusion matrices
  - Configure basic options (model, n‑grams) and run inference

6) Documentation & Safety (ongoing)
- Document dataset provenance, licensing, and known biases; add usage limitations.
- Expand README with examples, tips for large‑scale training, and performance notes.

Milestones
- v0.2.0: Improved metrics + saved predictions + expanded grids + logging.
- v0.3.0: CI + pre‑commit + config file + imbalance handling options.
- v0.4.0: Streamlit app for interactive demo and simple hosting.

---

## Streamlit App

- Run locally:
  - `pip install -r requirements.txt`
  - `streamlit run app/streamlit_app.py`
- Features:
  - Predict on single text or batch CSV using a saved pipeline
  - Train/evaluate from the UI with model, n‑grams, balancing and threshold options
  - Browse artifacts: metrics.json, predictions CSVs, EDA and confusion matrices
  - Auto-detects data in `data/` if present; otherwise uses project root
  - Defaults to saving/exploring artifacts under `outputs_streamlit/` (falls back to `outputs/`)

Tips
- If you see Matplotlib font cache warnings, set `MPLCONFIGDIR=$(pwd)/.mplconfig` and rerun.
- Loading legacy models referencing `src.preprocess` is supported in-app via a shim; if a model still fails, click Rerun or Clear cache and reselect the file.
