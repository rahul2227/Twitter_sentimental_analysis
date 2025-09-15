from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import joblib
import pandas as pd


def predict_texts(model_path: str, texts: List[str]) -> List[int]:
    pipeline = joblib.load(model_path)
    preds = pipeline.predict(texts)
    return list(map(int, preds))


def main():
    import argparse

    p = argparse.ArgumentParser(description='Run inference using saved pipeline.')
    p.add_argument('--model', default='outputs/artifacts/model_pipeline.joblib', help='Path to saved pipeline joblib')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', help='Single text to classify')
    group.add_argument('--csv', help='CSV file with a column named "tweet" to classify')
    p.add_argument('--out-csv', help='Where to save predictions CSV when using --csv')
    args = p.parse_args()

    if args.text:
        preds = predict_texts(args.model, [args.text])
        print(preds[0])
        return

    df = pd.read_csv(args.csv)
    if 'tweet' not in df.columns:
        print("CSV must contain a 'tweet' column", file=sys.stderr)
        sys.exit(2)
    preds = predict_texts(args.model, df['tweet'].astype(str).tolist())
    out_path = args.out_csv or (str(Path(args.csv).with_suffix('')) + '_preds.csv')
    out_df = df.copy()
    out_df['pred'] = preds
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == '__main__':
    main()

