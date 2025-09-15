import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import joblib
import threading

# Robust import of local package when running via Streamlit from different CWDs
try:  # pragma: no cover - runtime import convenience
    from twitter_sentiment.train import train_and_evaluate, online_train_and_evaluate
    from twitter_sentiment.evaluate import get_prediction_scores, compute_metrics
except ModuleNotFoundError:  # add repo root to sys.path and retry
    import sys as _sys
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    from twitter_sentiment.train import train_and_evaluate, online_train_and_evaluate
    from twitter_sentiment.evaluate import get_prediction_scores


# Ensure a writable MPL config dir for environments with restricted home dirs
MPL_DIR = Path('.mplconfig')
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_DIR.resolve()))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

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

    st.markdown('##### Re-save Loaded Model')
    save_col1, save_col2 = st.columns(2)
    with save_col1:
        target_out = st.text_input('Target outputs dir', value='outputs_resaved', key='resave_outputs')
    with save_col2:
        if st.button('Re-save model', key='btn_resave'):
            if pipeline is None:
                st.warning('Load a model first')
            else:
                try:
                    out_dir = Path(target_out)
                    (out_dir / 'artifacts').mkdir(parents=True, exist_ok=True)
                    joblib.dump(pipeline, out_dir / 'artifacts' / 'model_pipeline.joblib')
                    st.success(f'Re-saved to {out_dir}/artifacts/model_pipeline.joblib')
                except Exception as e:
                    st.exception(e)


def render_train_tab():
    st.subheader('Train & Evaluate')
    default_train = existing_path(['data/train.csv', 'train.csv'])
    default_test = existing_path(['data/test.csv', 'test.csv'])
    train_path = st.text_input('Train CSV path', value=default_train, key='train_csv_path')
    test_path = st.text_input('Test CSV path', value=default_test, key='test_csv_path')
    outputs = st.text_input('Outputs directory', value='outputs_streamlit', key='train_outputs_dir')

    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox('Model', options=['nb', 'cnb', 'logreg', 'logreg_l1', 'linearsvm', 'sgd', 'sgd_hinge', 'pa'], index=0)
        balance = st.selectbox('Balance', options=['none', 'downsample', 'upsample'], index=0)
    with c2:
        word_ngrams = st.text_input('Word n-grams', value='1,2', key='word_ngrams')
        char_ngrams = st.text_input('Char n-grams', value='3,5', key='char_ngrams')
    with c3:
        use_char = st.checkbox('Use char n-grams', value=False)
        calibrate = st.checkbox('Calibrate threshold', value=False)

    with st.expander('Preprocessing options', expanded=False):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            lowercase = st.checkbox('Lowercase', value=True, key='pp_lowercase')
            max_repeat = st.number_input('Max repeat chars', min_value=0, max_value=10, value=3, step=1, key='pp_max_repeat')
            hashtag_policy = st.selectbox('Hashtag policy', options=['strip', 'keep', 'segment'], index=0, key='pp_hashtag_policy')
            emoticons = st.checkbox('Map emoticons', value=True, key='pp_emoticons')
        with pc2:
            replace_url = st.checkbox('Replace URLs with <url>', value=True, key='pp_replace_url')
            replace_user = st.checkbox('Replace @user with <user>', value=True, key='pp_replace_user')
            number_policy = st.selectbox('Number policy', options=['keep', 'drop', 'token'], index=1, key='pp_number_policy')
            emoji_policy = st.selectbox('Emoji policy', options=['ignore', 'drop', 'map'], index=0, key='pp_emoji_policy')
        with pc3:
            contractions = st.selectbox('Contractions', options=['none', 'default'], index=0, key='pp_contractions')
            negation_scope = st.number_input('Negation scope (tokens)', min_value=0, max_value=10, value=0, step=1, key='pp_neg_scope')

    c4, c5, c6 = st.columns(3)
    with c4:
        test_size = st.number_input('Validation size', min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    with c5:
        seed = st.number_input('Seed', min_value=0, max_value=10_000, value=42, step=1)
    with c6:
        use_grid = st.checkbox('Use small grid search', value=True)

    c7, c8 = st.columns(2)
    with c7:
        select_k = st.number_input('SelectKBest chi² (0 = off)', min_value=0, max_value=200000, value=0, step=1000, key='select_k')
    with c8:
        calibrate_probs = st.checkbox('Calibrate probabilities (sigmoid)', value=False, key='calibrate_probs')

    c9, c10 = st.columns(2)
    with c9:
        use_hashing = st.checkbox('Use hashing vectorizer', value=False, key='use_hashing')
    with c10:
        hashing_features = st.number_input('Hashing features', min_value=2**16, max_value=2**22, value=2**20, step=2**18, key='hashing_features')

    resume_ckpt = st.checkbox('Resume from checkpoint (if exists)', value=False, key='resume_offline')

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
                        select_k=int(select_k),
                        calibrate_probabilities=bool(calibrate_probs),
                        balance=balance,
                        calibrate_threshold=calibrate,
                        resume=bool(resume_ckpt),
                        use_hashing=bool(use_hashing),
                        hashing_features=int(hashing_features),
                        lowercase=bool(lowercase),
                        max_repeat=int(max_repeat),
                        hashtag_policy=str(hashtag_policy),
                        replace_url=bool(replace_url),
                        replace_user=bool(replace_user),
                        number_policy=str(number_policy),
                        emoji_policy=str(emoji_policy),
                        use_emoticons=bool(emoticons),
                        contractions=str(contractions),
                        negation_scope=int(negation_scope),
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
                st.image(str(p), caption=name, width='stretch')

    # Interactive: Threshold explorer (if plotly and val scores available)
    if HAS_PLOTLY:
        val_csv = reports_dir / 'val_predictions.csv'
        if val_csv.exists():
            try:
                dfv = pd.read_csv(val_csv)
                if 'score' in dfv.columns and 'y_true' in dfv.columns:
                    st.markdown('##### Threshold explorer (validation set)')
                    lo, hi = float(dfv['score'].min()), float(dfv['score'].max())
                    default_t = 0.5
                    met = None
                    if metrics_path.exists():
                        try:
                            met = json.loads(metrics_path.read_text())
                            default_t = float(met.get('calibrated_threshold') or default_t)
                        except Exception:
                            pass
                    t = st.slider('Decision threshold', min_value=lo, max_value=hi, value=min(max(default_t, lo), hi), step=(hi - lo) / 100 if hi > lo else 0.01, key='thresh_slider')
                    y_pred = (dfv['score'].values >= t).astype(int)
                    y_true = dfv['y_true'].values
                    m = compute_metrics(y_true, y_pred, scores=dfv['score'].values)
                    st.write({k: v for k, v in m.items() if k in ['accuracy', 'f1_macro', 'roc_auc']})
                    cm = m['confusion_matrix']
                    fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0', 'Pred 1'], y=['True 0', 'True 1'], colorscale='Blues', showscale=True))
                    fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.info(f'Interactive threshold explorer unavailable: {e}')

    # Offline training curve (if present)
    curve_path = reports_dir / 'training_curve.json'
    if curve_path.exists():
        try:
            st.markdown('##### Offline Training Curve')
            curve = json.loads(curve_path.read_text())
            dfc = pd.DataFrame({
                'fraction': curve.get('fractions', []),
                'train_loss': curve.get('train_loss', []),
                'val_loss': curve.get('val_loss', []),
            })
            if HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfc['fraction'], y=dfc['train_loss'], mode='lines+markers', name='train'))
                fig.add_trace(go.Scatter(x=dfc['fraction'], y=dfc['val_loss'], mode='lines+markers', name='val'))
                fig.update_layout(xaxis_title='Train fraction', yaxis_title='Loss', height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, width='stretch')
            else:
                st.line_chart(dfc.set_index('fraction'))
        except Exception as e:
            st.info(f'Offline training curve unavailable: {e}')

    # Top features (interactive)
    tf_path = reports_dir / 'top_features.json'
    if tf_path.exists() and HAS_PLOTLY:
        try:
            top = json.loads(tf_path.read_text())
            st.markdown('##### Top Features')
            colp, coln = st.columns(2)
            pos = pd.DataFrame(top.get('top_positive', []))
            neg = pd.DataFrame(top.get('top_negative', []))
            if not pos.empty:
                with colp:
                    figp = px.bar(pos.iloc[::-1], x='weight', y='feature', orientation='h', title='Top Positive')
                    st.plotly_chart(figp, width='stretch')
            if not neg.empty:
                with coln:
                    fign = px.bar(neg, x='weight', y='feature', orientation='h', title='Top Negative')
                    st.plotly_chart(fign, width='stretch')
        except Exception as e:
            st.info(f'Top features unavailable: {e}')

    # Error analysis: misclassification browser
    if (reports_dir / 'val_predictions.csv').exists():
        try:
            dfv = pd.read_csv(reports_dir / 'val_predictions.csv')
            if 'y_true' in dfv.columns:
                st.markdown('##### Misclassification Browser (validation)')
                f1, f2, f3 = st.columns(3)
                with f1:
                    tl = st.selectbox('True label', options=['all', '0', '1'], index=0, key='mb_true')
                with f2:
                    pl = st.selectbox('Pred label', options=['all', '0', '1'], index=0, key='mb_pred')
                with f3:
                    if 'score' in dfv.columns:
                        smin = float(dfv['score'].min()); smax = float(dfv['score'].max())
                        s_lo, s_hi = st.slider('Score range', min_value=smin, max_value=smax, value=(smin, smax), key='mb_score')
                    else:
                        s_lo, s_hi = None, None

                view = dfv.copy()
                if tl in {'0','1'}:
                    view = view[view['y_true'] == int(tl)]
                if pl in {'0','1'}:
                    view = view[view['y_pred'] == int(pl)]
                if s_lo is not None:
                    view = view[(view['score'] >= s_lo) & (view['score'] <= s_hi)]
                st.dataframe(view.head(200))
        except Exception as e:
            st.info(f'Misclassification browser unavailable: {e}')


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

    # Live training (streaming) tab
    st.markdown('---')
    st.header('Live Training (Streaming)')
    with st.expander('Run online training with partial_fit', expanded=False):
        lt_col1, lt_col2 = st.columns(2)
        with lt_col1:
            lt_train = existing_path(['data/train.csv', 'train.csv'])
            lt_train_path = st.text_input('Train CSV path (online)', value=lt_train, key='lt_train_path')
            lt_outputs = st.text_input('Outputs directory (online)', value='outputs_live', key='lt_outputs')
            lt_model = st.selectbox('Model (online)', options=['nb', 'cnb', 'sgd', 'sgd_hinge', 'pa'], index=0, key='lt_model')
            lt_batch = st.number_input('Batch size', min_value=256, max_value=50000, value=4096, step=256, key='lt_batch')
            lt_epochs = st.number_input('Epochs', min_value=1, max_value=50, value=1, step=1, key='lt_epochs')
        with lt_col2:
            lt_eval_every = st.number_input('Eval every N batches', min_value=1, max_value=1000, value=1, step=1, key='lt_eval_every')
            lt_ckpt_every = st.number_input('Checkpoint every N batches', min_value=1, max_value=1000, value=5, step=1, key='lt_ckpt_every')
            lt_word_ngrams = st.text_input('Word n-grams', value='1,2', key='lt_word_ngrams')
            lt_char_ngrams = st.text_input('Char n-grams', value='3,5', key='lt_char_ngrams')
            lt_use_char = st.checkbox('Use char n-grams', value=False, key='lt_use_char')
            lt_use_hashing = st.checkbox('Use hashing vectorizer', value=False, key='lt_use_hashing')
            lt_hashing_features = st.number_input('Hashing features (online)', min_value=2**16, max_value=2**22, value=2**20, step=2**18, key='lt_hashing_features')

        with st.expander('Preprocessing options (online)', expanded=False):
            lpc1, lpc2, lpc3 = st.columns(3)
            with lpc1:
                lt_lowercase = st.checkbox('Lowercase', value=True, key='lt_lowercase')
                lt_max_repeat = st.number_input('Max repeat chars', min_value=0, max_value=10, value=3, step=1, key='lt_max_repeat')
                lt_hashtag_policy = st.selectbox('Hashtag policy', options=['strip', 'keep', 'segment'], index=0, key='lt_hashtag_policy')
                lt_emoticons = st.checkbox('Map emoticons', value=True, key='lt_emoticons')
            with lpc2:
                lt_replace_url = st.checkbox('Replace URLs with <url>', value=True, key='lt_replace_url')
                lt_replace_user = st.checkbox('Replace @user with <user>', value=True, key='lt_replace_user')
                lt_number_policy = st.selectbox('Number policy', options=['keep', 'drop', 'token'], index=1, key='lt_number_policy')
                lt_emoji_policy = st.selectbox('Emoji policy', options=['ignore', 'drop', 'map'], index=0, key='lt_emoji_policy')
            with lpc3:
                lt_contractions = st.selectbox('Contractions', options=['none', 'default'], index=0, key='lt_contractions')
                lt_neg_scope = st.number_input('Negation scope (tokens)', min_value=0, max_value=10, value=0, step=1, key='lt_neg_scope')

        # Background training state
        if 'lt_thread' not in st.session_state:
            st.session_state['lt_thread'] = None
            st.session_state['lt_stop'] = None

        colb1, colb2, colb3, colb4 = st.columns(4)
        if colb1.button('Start/Resume live training'):
            if not Path(lt_train_path).exists():
                st.error('Train CSV not found')
            else:
                if st.session_state['lt_thread'] and st.session_state['lt_thread'].is_alive():
                    st.info('Live training already running...')
                else:
                    stop_event = threading.Event()
                    st.session_state['lt_stop'] = stop_event
                    args = dict(
                        train_csv=lt_train_path,
                        outputs_dir=lt_outputs,
                        model=lt_model,
                        use_char_ngrams=lt_use_char,
                        word_ngrams=lt_word_ngrams,
                        char_ngrams=lt_char_ngrams,
                        use_hashing=lt_use_hashing,
                        hashing_features=int(lt_hashing_features),
                        random_state=42,
                        batch_size=int(lt_batch),
                        epochs=int(lt_epochs),
                        eval_every=int(lt_eval_every),
                        checkpoint_every=int(lt_ckpt_every),
                        lowercase=bool(lt_lowercase),
                        max_repeat=int(lt_max_repeat),
                        hashtag_policy=str(lt_hashtag_policy),
                        replace_url=bool(lt_replace_url),
                        replace_user=bool(lt_replace_user),
                        number_policy=str(lt_number_policy),
                        emoji_policy=str(lt_emoji_policy),
                        use_emoticons=bool(lt_emoticons),
                        contractions=str(lt_contractions),
                        negation_scope=int(lt_neg_scope),
                        stop_event=stop_event,
                        resume=True,
                    )
                    th = threading.Thread(target=online_train_and_evaluate, kwargs=args, daemon=True)
                    th.start()
                    st.session_state['lt_thread'] = th
                    st.success('Started live training in background')

        if colb2.button('Stop live training'):
            evt = st.session_state.get('lt_stop')
            if evt is not None:
                evt.set()
                st.info('Stop signal sent')
            else:
                st.info('No live training to stop')

        # Auto-refresh controls (stateful, no full page reload)
        with colb3:
            auto = st.checkbox('Auto-refresh metrics', value=False, key='lt_auto_refresh')
            every = st.number_input('every (sec)', min_value=1, max_value=300, value=10, step=1, key='lt_auto_sec')
            if auto:
                now = time.time()
                next_key = 'lt_next_auto'
                if next_key not in st.session_state:
                    st.session_state[next_key] = now + float(every)
                elif now >= st.session_state[next_key]:
                    st.session_state[next_key] = now + float(every)
                    try:
                        st.rerun()
                    except Exception:
                        pass
                else:
                    remaining = int(st.session_state[next_key] - now)
                    st.caption(f'Auto-refresh in ~{remaining}s')

        if colb4.button('Refresh live metrics'):
            try:
                st.rerun()
            except Exception:
                st.info('Please use the Rerun button in Streamlit menu')

        # Live metrics view
        live_reports = Path(lt_outputs) / 'reports'
        live_metrics_path = live_reports / 'metrics_online.json'
        if live_metrics_path.exists():
            try:
                mj = json.loads(live_metrics_path.read_text())
                st.markdown('##### Live metrics (validation)')
                st.json({k: mj.get(k) for k in ['epoch', 'batch', 'samples_seen', 'total_rows']})
                if HAS_PLOTLY and 'validation' in mj and 'confusion_matrix' in mj['validation']:
                    cm = mj['validation']['confusion_matrix']
                    fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0', 'Pred 1'], y=['True 0', 'True 1'], colorscale='Blues'))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, width='stretch')
                # Live training loss curve (if provided)
                if HAS_PLOTLY and isinstance(mj.get('train_curve'), list) and mj['train_curve']:
                    st.markdown('##### Live Training Loss Curve')
                    dfc = pd.DataFrame(mj['train_curve'])
                    figlc = px.line(dfc, x='step', y='loss')
                    figlc.update_layout(yaxis_title='Loss', xaxis_title='Samples seen', height=300, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(figlc, width='stretch')
            except Exception as e:
                st.info(f'Live metrics not available: {e}')


if __name__ == '__main__':
    main()
