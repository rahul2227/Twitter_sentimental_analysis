import io
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import joblib

# Robust import of local package when running via Streamlit from different CWDs
try:  # pragma: no cover - runtime import convenience
    from twitter_sentiment.train import train_and_evaluate
    from twitter_sentiment.evaluate import get_prediction_scores
except ModuleNotFoundError:  # add repo root to sys.path and retry
    import sys as _sys
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    from twitter_sentiment.train import train_and_evaluate
    from twitter_sentiment.evaluate import get_prediction_scores


# Ensure a writable MPL config dir for environments with restricted home dirs
MPL_DIR = Path('.mplconfig')
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_DIR.resolve()))


@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: str):
    """Load a saved sklearn pipeline with legacy compatibility.

    Some older artifacts were trained when the project lived under a
    'src' package. Unpickling may fail with ModuleNotFoundError: 'src'.
    We install a small shim to map 'src.preprocess' to the current
    twitter_sentiment.preprocess module and retry.
    """
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        msg = str(e)
        if "No module named 'src'" in msg or msg.strip() == "No module named 'src'":
            # Install a minimal shim for legacy artifacts
            import sys as _sys
            import types as _types
            from twitter_sentiment import preprocess as _ts_pre

            # Create a top-level 'src' module if missing
            if 'src' not in _sys.modules:
                _sys.modules['src'] = _types.ModuleType('src')

            # Map 'src.preprocess' to current module
            _sys.modules['src.preprocess'] = _ts_pre

            # Also expose common names on the shim for safety
            setattr(_sys.modules['src'], 'preprocess', _ts_pre)
            for _name in ['preprocess_and_tokenize', 'tokenize', 'tokenize_tweet']:
                if not hasattr(_ts_pre, _name) and _name != 'preprocess_and_tokenize':
                    # no-op alias when name doesn't exist in new module
                    continue
                setattr(_sys.modules['src'], _name, getattr(_ts_pre, 'preprocess_and_tokenize', None))
            # Retry load
            return joblib.load(model_path)
        raise


def existing_path(defaults: list[str]) -> str:
    for p in defaults:
        if Path(p).exists():
            return p
    return defaults[0]


def render_predict_tab():
    st.subheader('Predict')
    default_model = existing_path([
        'outputs_streamlit/artifacts/model_pipeline.joblib',
        'outputs/artifacts/model_pipeline.joblib',
    ])
    model_path = st.text_input('Model pipeline path (.joblib)', value=default_model, key='predict_model_path')

    pipeline = None
    if model_path and Path(model_path).exists():
        try:
            pipeline = load_pipeline(model_path)
            st.success('Model loaded successfully')
        except Exception as e:
            st.error(f'Failed to load model: {e}')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('##### Single text')
        text = st.text_area('Enter text', height=120, key='single_text')
        if st.button('Predict', type='primary', key='predict_single'):
            if not pipeline:
                st.warning('Load a model first')
            elif not text.strip():
                st.warning('Enter some text')
            else:
                pred = int(pipeline.predict([text])[0])
                scores = get_prediction_scores(pipeline, [text])
                score = None if scores is None else float(scores[0])
                st.write(f'Prediction: {pred}')
                if score is not None:
                    st.write(f'Score: {score:.4f}')

    with col2:
        st.markdown('##### CSV batch')
        up = st.file_uploader('Upload CSV with a "tweet" column', type=['csv'])
        if up is not None and pipeline:
            df = pd.read_csv(up)
            if 'tweet' not in df.columns:
                st.error("CSV must contain a 'tweet' column")
            else:
                preds = pipeline.predict(df['tweet'].astype(str).tolist())
                scores = get_prediction_scores(pipeline, df['tweet'].astype(str).tolist())
                out = df.copy()
                out['pred'] = preds
                if scores is not None:
                    out['score'] = scores
                csv_bytes = out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    'Download predictions CSV',
                    data=csv_bytes,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
                st.dataframe(out.head(20))


def render_train_tab():
    st.subheader('Train & Evaluate')
    default_train = existing_path(['data/train.csv', 'train.csv'])
    default_test = existing_path(['data/test.csv', 'test.csv'])
    train_path = st.text_input('Train CSV path', value=default_train, key='train_csv_path')
    test_path = st.text_input('Test CSV path', value=default_test, key='test_csv_path')
    outputs = st.text_input('Outputs directory', value='outputs_streamlit', key='train_outputs_dir')

    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox('Model', options=['nb', 'logreg', 'linearsvm'], index=0)
        balance = st.selectbox('Balance', options=['none', 'downsample', 'upsample'], index=0)
    with c2:
        word_ngrams = st.text_input('Word n-grams', value='1,2', key='word_ngrams')
        char_ngrams = st.text_input('Char n-grams', value='3,5', key='char_ngrams')
    with c3:
        use_char = st.checkbox('Use char n-grams', value=False)
        calibrate = st.checkbox('Calibrate threshold', value=False)

    c4, c5, c6 = st.columns(3)
    with c4:
        test_size = st.number_input('Validation size', min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    with c5:
        seed = st.number_input('Seed', min_value=0, max_value=10_000, value=42, step=1)
    with c6:
        use_grid = st.checkbox('Use small grid search', value=True)

    if st.button('Run training', type='primary'):
        if not Path(train_path).exists() or not Path(test_path).exists():
            st.error('Train or Test CSV not found')
        else:
            with st.spinner('Training...'):
                try:
                    train_and_evaluate(
                        train_csv=train_path,
                        test_csv=test_path,
                        outputs_dir=outputs,
                        model=model,
                        use_char_ngrams=use_char,
                        word_ngrams=word_ngrams,
                        char_ngrams=char_ngrams,
                        test_size=float(test_size),
                        random_state=int(seed),
                        do_grid=bool(use_grid),
                        balance=balance,
                        calibrate_threshold=calibrate,
                    )
                    st.success('Training completed')
                except Exception as e:
                    st.exception(e)


def render_artifacts_tab():
    st.subheader('Artifacts & Reports')
    default_outputs = existing_path(['outputs_streamlit', 'outputs'])
    out_dir = st.text_input('Outputs directory', value=default_outputs, key='artifacts_outputs_dir')
    reports_dir = Path(out_dir) / 'reports'
    figures_dir = Path(out_dir) / 'figures'

    # Metrics JSON
    metrics_path = reports_dir / 'metrics.json'
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            st.markdown('##### metrics.json')
            st.json(data)
        except Exception as e:
            st.error(f'Failed to read metrics.json: {e}')
    else:
        st.info('metrics.json not found')

    # Predictions CSVs
    for name in ['val_predictions.csv', 'test_predictions.csv']:
        p = reports_dir / name
        if p.exists():
            st.markdown(f'##### {name}')
            try:
                df = pd.read_csv(p)
                st.dataframe(df.head(20))
                st.download_button(
                    f'Download {name}',
                    data=p.read_bytes(),
                    file_name=name,
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f'Failed to load {name}: {e}')

    # Figures
    img_names = [
        'barplot_label_length.png',
        'label_counts.png',
        'confusion_matrix_val.png',
        'confusion_matrix_test.png',
    ]
    cols = st.columns(2)
    for i, name in enumerate(img_names):
        p = figures_dir / name
        if p.exists():
            with cols[i % 2]:
                st.image(str(p), caption=name, use_column_width=True)


def main():
    st.set_page_config(page_title='Twitter Sentiment', layout='wide')
    st.title('Twitter Sentiment Analysis')
    st.caption('Train models, run predictions, and inspect artifacts')

    tabs = st.tabs(['Predict', 'Train/Evaluate', 'Artifacts'])
    with tabs[0]:
        render_predict_tab()
    with tabs[1]:
        render_train_tab()
    with tabs[2]:
        render_artifacts_tab()


if __name__ == '__main__':
    main()
