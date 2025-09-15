## Twitter Sentiment Analysis — End‑to‑End, Modular, and Interactive

A modern, modular Twitter sentiment analysis project with:
- Fast, configurable preprocessing (no external corpora)
- Solid, extensible modeling options (linear, NB, streaming‑friendly)
- A clean CLI for training, evaluation, prediction (offline + online)
- An interactive Streamlit app for training, live monitoring, error analysis, and visualizations
- Reproducible artifacts (pipelines, metrics, curves), and unit tests

This README consolidates all developer and user guidance for the project.

---

## Contents
- Overview and Features
- Project Structure
- Setup and Requirements
- Data Schema
- Preprocessing (configurable)
- Modeling Options and Feature Selection
- Offline Training and Outputs
- Online/Streaming Training
- Streamlit App (interactive)
- Resume Training and Re‑save Legacy Models
- Testing
- Troubleshooting
- Roadmap

---

## Overview and Features

- Tokenization and normalization are handled by a configurable, self‑contained regex pipeline (no NLTK corpora required). Placeholders for `<url>` and `<user>`, hashtag handling, emoji/emoticon mapping, repeat‑char control, contractions, and optional negation scope.
- Vectorization via TF‑IDF over word n‑grams (optional char n‑grams). Optional HashingVectorizer for streaming (fixed dimensionality).
- Models: MultinomialNB, ComplementNB, Logistic Regression (L2/L1), LinearSVC, SGD (hinge/log), Passive‑Aggressive. Optional calibrated probabilities.
- CLI: Train/evaluate (offline), predict (single text or CSV), and online/streaming training with partial_fit.
- Streamlit app: Predict, train/evaluate, browse artifacts, threshold explorer, top‑features explorer, misclassification browser, offline training curve, and live training (background thread with auto‑refresh).
- Artifacts: sklearn pipeline, metrics JSONs, EDA/confusion matrix figures, top features (when supported), training curves.
- Unit tests: core preprocessing, data loading, metric sanity checks, and modeling sanity tests.

---

## Project Structure

- `twitter_sentiment/` — source package
  - `preprocess.py` — Tokenizer + PreprocessorConfig; composable normalization.
  - `model.py` — vectorizer builders (Count/Hashing + TF‑IDF), models, and grids.
  - `data.py` — dataset IO, outputs directory creation, EDA plots.
  - `evaluate.py` — metrics, confusion matrix saving, top features, loss calculation.
  - `train.py` — CLI entry; offline and online training; resume; artifacts.
  - `predict.py` — CLI inference on text/CSV.
- `app/streamlit_app.py` — Streamlit app (predict, train/evaluate, artifacts, live training).
- `tests/` — unit tests.
- `data/` — place `train.csv`, `test.csv` here (preferred).
- `outputs/` — default artifacts (created on first run).

---

## Setup and Requirements

- Python 3.9–3.11 recommended; project validated in conda env `test_env`.
- Install: `pip install -r requirements.txt` (includes pandas, numpy, scikit‑learn, matplotlib, seaborn, streamlit, plotly).
- Headless environments: set `MPLCONFIGDIR` to writable dir (to silence font cache warnings):
  - `export MPLCONFIGDIR=$(pwd)/.mplconfig`
- Git hygiene: `.gitignore` excludes generated outputs, caches, IDE files. Untrack previously committed artifacts with, e.g., `git rm -r --cached outputs .mplconfig __pycache__ .idea`.

Conda examples
- `conda run -n test_env python -m unittest -v`
- `conda run -n test_env streamlit run app/streamlit_app.py`

---

## Data Schema

- Train CSV: `label` (0/1) and `tweet` (string) required.
- Test CSV: `tweet` required; `label` optional (metrics computed if present).
- Preferred location: `data/train.csv`, `data/test.csv` (auto‑detected by CLI/app).

---

## Preprocessing (configurable)

PreprocessorConfig (selected highlights):
- `lowercase` (default True)
- `max_repeat_char` (collapse long repeats)
- `hashtag_policy`: `strip|keep|segment`
- `replace_url`, `replace_user` (placeholders `<url>`, `<user>`)
- `number_policy`: `keep|drop|token`
- `emoji_policy`: `ignore|drop|map` + basic emoticon mapping
- `contractions`: `none|default` or custom dict
- `negation_scope`: integer to mark next N tokens with `_neg`

CLI flags (examples):
- `--no-lowercase`, `--max-repeat 3`, `--hashtag-policy strip`,
- `--no-replace-url`, `--no-replace-user`,
- `--number-policy drop`, `--emoji-policy ignore`, `--no-emoticons`,
- `--contractions default`, `--negation-scope 0`

Vectorizer options:
- `--use-char-ngrams`, `--word-ngrams 1,2`, `--char-ngrams 3,5`
- `--use-hashing` (HashingVectorizer), `--hashing-features 1048576`

---

## Modeling Options and Feature Selection

Models:
- `nb` (MultinomialNB), `cnb` (ComplementNB)
- `logreg` (L2), `logreg_l1` (L1/liblinear)
- `linearsvm` (LinearSVC)
- `sgd` (log_loss), `sgd_hinge`, `pa` (PassiveAggressive)

Extras:
- `--select-k N` — optional `SelectKBest(chi²)` feature selection (cap features for speed/stability).
- `--calibrate-probabilities` — wraps model with `CalibratedClassifierCV` (for probabilities and threshold tuning).

Notes:
- Skip grid search when calibrating probabilities (avoids nested CV).
- For very large `select_k`, prefer smaller values or linear models without calibration.

---

## Offline Training and Outputs

Train and evaluate (offline):
```
python -m twitter_sentiment.train \
  --train data/train.csv \
  --test data/test.csv \
  --outputs outputs \
  --model nb              # or: cnb, logreg, logreg_l1, linearsvm, sgd, sgd_hinge, pa \
  --no-grid               # optional: disable grid search
```

Logging and progress:
- `--log-level INFO|DEBUG|...` to control verbosity (default INFO).
- `--no-progress` disables extra verbosity (incl. tqdm where applicable).
  - Grid search prints fold/candidate progress when progress is enabled.

Key outputs under `outputs/`:
- `artifacts/model_pipeline.joblib` — trained sklearn pipeline
- `reports/metrics.json` — validation metrics (and test metrics if labels exist)
- `reports/val_predictions.csv`, `reports/test_predictions.csv`
- `reports/top_features.json` — top positive/negative tokens (linear/NB models)
- `reports/training_curve.json` — offline loss curves (train/val vs fraction)
- `figures/` — EDA plots + confusion matrices

Sample figures (paths after a run):
- `![Label counts](outputs/figures/label_counts.png)`
- `![Confusion matrix (val)](outputs/figures/confusion_matrix_val.png)`

---

## Online / Streaming Training

Incremental training with `partial_fit` and chunked CSV reading.

CLI example:
```
python -m twitter_sentiment.train \
  --online \
  --train data/train.csv \
  --outputs outputs_live \
  --model nb \           # or: cnb, sgd, sgd_hinge, pa
  --batch-size 4096 \     # chunksize for streaming
  --epochs 1 \            # passes over data
  --eval-every 1 \        # evaluate validation every N batches
  --checkpoint-every 5    # save pipeline every N batches
# streaming‑friendly vectorizer:
# --use-hashing --hashing-features 1048576
```

Outputs under `outputs_live/`:
- `artifacts/model_pipeline.joblib` — periodically checkpointed
- `reports/metrics_online.json` — rolling metrics + `train_curve` (samples seen vs loss)

Progress bars:
- Online mode displays a tqdm bar over seen samples (per epoch) with loss and periodic validation metrics in the postfix.
- Disable with `--no-progress`.

---

## Streamlit App (interactive)

Run locally:
- `pip install -r requirements.txt`
- `streamlit run app/streamlit_app.py`

Features:
- Predict (single text + CSV). Re‑save loaded model to new outputs dir.
- Train/Evaluate (offline): configure model, n‑grams, preprocessing, selection, calibration, and Resume from checkpoint.
- Artifacts: browse metrics.json, predictions, figures; interactive threshold explorer; top‑features explorer; misclassification browser; offline training loss curves.
- Live Training (streaming): background thread with Start/Resume and Stop; hashing option; auto‑refresh metrics (state‑preserving); live confusion matrix and loss curve.

Tips:
- The app uses `width='stretch'` for charts/images to avoid deprecation warnings.
- If Matplotlib font cache warnings appear, set `MPLCONFIGDIR=$(pwd)/.mplconfig`.

---

## Resume Training and Re‑save Legacy Models

Resume (offline):
- Add `--resume` (CLI) or enable “Resume from checkpoint” in the app.
- Works best with partial_fit models (NB/SGD/PA); others refit.

Re‑save legacy models:
- Predict tab → “Re‑save Loaded Model” to write the currently loaded pipeline to `target/outputs/artifacts/model_pipeline.joblib`.
- Legacy models from older layouts are supported by a small shim that maps `src.preprocess` → `twitter_sentiment.preprocess` during load.

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

- Matplotlib font cache warnings: set `MPLCONFIGDIR=$(pwd)/.mplconfig`.
- Legacy artifacts: load in the app and re‑save; or add a small `sys.modules['src.preprocess']` shim before joblib.load.
- Sklearn warnings: with a custom tokenizer, scikit‑learn notes `token_pattern` is ignored — expected and safe.
- For best Streamlit performance, consider installing `watchdog`.

---

## Roadmap (high‑level)

- Evaluation & Reporting: expand metrics JSON, save more diagnostics.
- Modeling: richer grids, cross‑validation, imbalance strategies.
- Preprocessing: optional emoji tables, configurable placeholder/limits.
- Developer experience: logging, config files, pre‑commit, CI.
- App: deeper error analysis, richer model introspection, export utilities.
