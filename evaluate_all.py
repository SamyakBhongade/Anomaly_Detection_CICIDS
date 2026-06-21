"""
evaluate_all.py
===============
Run full evaluation on all saved models for both NSL-KDD and CICIDS2017.
Loads saved models from disk — no re-training needed.

Usage:
    python evaluate_all.py               # evaluate both datasets
    python evaluate_all.py --dataset cicids
    python evaluate_all.py --dataset nslkdd
"""

import os, sys, argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR  = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
SAVE_DIR_NSL= os.path.join(PROJECT_ROOT, 'data', 'processed', 'nslkdd')
SAVE_DIR_CIC= os.path.join(PROJECT_ROOT, 'data', 'processed', 'cicids')

NSL_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'NSLKDD', 'KDDTrain+.txt')
NSL_TEST_PATH  = os.path.join(PROJECT_ROOT, 'data', 'NSLKDD', 'KDDTest+.txt')
CICIDS_DIR     = os.path.join(PROJECT_ROOT, 'data', 'CICIDS2017')


# ── Load processed data from .npz cache ──────────────────────────────────────

def load_cached(dataset):
    d = SAVE_DIR_NSL if dataset == 'nslkdd' else SAVE_DIR_CIC
    f = os.path.join(d, f'{dataset}_arrays.npz')

    if os.path.exists(f):
        data   = np.load(f)
        X_test = data['X_test'].astype(np.float32)
        y_test = data['y_test'].astype(int)
        print(f"[Data] {dataset.upper()} loaded from cache: {X_test.shape}")
        return X_test, y_test

    # Cache missing — try to preprocess from raw files (NSL-KDD only)
    if dataset == 'nslkdd':
        if not os.path.exists(NSL_TRAIN_PATH):
            raise FileNotFoundError(
                f"NSL cache not found and raw data missing at {NSL_TRAIN_PATH}"
            )
        print("[Data] NSL-KDD cache missing — preprocessing from raw files...")
        from preprocess import prepare_nslkdd
        X_train, X_test, y_train, y_test, _ = prepare_nslkdd(
            train_path=NSL_TRAIN_PATH, test_path=NSL_TEST_PATH,
            save_dir=SAVE_DIR_NSL, use_smote=True, random_state=42
        )
        os.makedirs(SAVE_DIR_NSL, exist_ok=True)
        np.savez_compressed(f, X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)
        print(f"[Data] NSL-KDD cached -> {f}")
        return X_test.astype(np.float32), y_test.astype(int)

    raise FileNotFoundError(
        f"Cached arrays not found: {f}\n"
        f"Run  python train_local.py --dataset {dataset}  first."
    )


# ── Load saved models ─────────────────────────────────────────────────────────

def load_models(dataset):
    import tensorflow as tf
    from transformer_model import TransformerBlock
    from vae_model         import VAELoss, sampling
    from ensemble          import EnsembleMetaClassifier

    tag = dataset   # 'cicids' or 'nslkdd' (matches saved file names)
    print(f"\n[Load] Loading all models for {dataset.upper()}...")

    bilstm = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, f'bilstm_{tag}.keras'))
    cnn    = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, f'cnn_{tag}.keras'))
    transformer = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, f'transformer_{tag}.keras'),
        custom_objects={'TransformerBlock': TransformerBlock})

    vae_dir    = os.path.join(MODELS_DIR, f'vae_{tag}')
    custom_vae = {'VAELoss': VAELoss, 'sampling': sampling}
    encoder = tf.keras.models.load_model(
        os.path.join(vae_dir, 'vae_encoder.keras'),
        custom_objects=custom_vae)
    decoder = tf.keras.models.load_model(
        os.path.join(vae_dir, 'vae_decoder.keras'),
        custom_objects=custom_vae)

    ens_dir = os.path.join(MODELS_DIR, f'ensemble_{tag}')
    ens = EnsembleMetaClassifier()
    ens.load(ens_dir)

    print(f"[Load] All models loaded for {dataset.upper()}")
    return bilstm, cnn, transformer, encoder, decoder, ens


# ── Run evaluation ────────────────────────────────────────────────────────────

def evaluate_dataset(dataset):
    from bilstm_model      import predict_proba_bilstm
    from cnn_model         import predict_proba_cnn
    from transformer_model import predict_proba_transformer
    from vae_model         import predict_proba_vae
    from ensemble          import collect_probabilities
    from evaluate          import (compute_all_metrics, plot_confusion_matrix,
                                   plot_roc_curves, plot_metrics_bar,
                                   save_metrics_table)

    X_test, y_test = load_cached(dataset)
    bilstm, cnn, transformer, enc, dec, ens = load_models(dataset)

    out_dir = os.path.join(OUTPUTS_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)
    tag = dataset.upper()

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {tag}")
    print(f"{'='*60}")

    # Individual predictions
    print("\n[Eval] Running individual model predictions...")
    p_bilstm      = predict_proba_bilstm(bilstm, X_test)
    p_cnn         = predict_proba_cnn(cnn, X_test)
    p_transformer = predict_proba_transformer(transformer, X_test)
    p_vae         = predict_proba_vae(enc, dec, X_test)

    # Ensemble
    proba_matrix = collect_probabilities(bilstm, cnn, transformer, enc, dec, X_test)
    p_ens  = ens.predict_proba(proba_matrix)
    y_pred = ens.predict(proba_matrix)

    # Compute metrics for all models
    results = []
    for name, proba in [('BiLSTM',      p_bilstm),
                        ('CNN',          p_cnn),
                        ('Transformer',  p_transformer),
                        ('VAE',          p_vae)]:
        preds = (proba >= 0.5).astype(int)
        m = compute_all_metrics(y_test, preds, proba, name)
        plot_confusion_matrix(y_test, preds, name, out_dir)
        results.append(m)

    # Ensemble metrics
    m_ens = compute_all_metrics(y_test, y_pred, p_ens, 'Ensemble (Ours)')
    plot_confusion_matrix(y_test, y_pred, 'Ensemble (Ours)', out_dir)
    results.append(m_ens)

    # Plots
    plot_roc_curves(
        [{'name': 'BiLSTM',       'proba': p_bilstm},
         {'name': 'CNN',           'proba': p_cnn},
         {'name': 'Transformer',   'proba': p_transformer},
         {'name': 'VAE',           'proba': p_vae},
         {'name': 'Ensemble',      'proba': p_ens}],
        y_test, out_dir, tag
    )
    plot_metrics_bar(results, out_dir, tag)
    save_metrics_table(results, out_dir, tag)

    return results


# ── Final combined summary ────────────────────────────────────────────────────

def print_final_summary(all_results: dict):
    print(f"\n\n{'#'*65}")
    print("  FINAL PROJECT EVALUATION SUMMARY")
    print(f"{'#'*65}")

    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'FPR']

    for dataset, results in all_results.items():
        df = pd.DataFrame(results)[cols]
        print(f"\n  Dataset: {dataset.upper()}")
        print("  " + "-"*60)
        print(df.to_string(index=False))
        print()

    # Highlight the ensemble row from each dataset
    print(f"\n{'='*65}")
    print("  ENSEMBLE (OURS) — CROSS-DATASET COMPARISON")
    print(f"{'='*65}")
    rows = []
    for dataset, results in all_results.items():
        ens_row = next((r for r in results if 'Ensemble' in r['Model']), None)
        if ens_row:
            row = {'Dataset': dataset.upper()}
            row.update({k: ens_row[k] for k in cols if k != 'Model'})
            rows.append(row)
    df_ens = pd.DataFrame(rows)
    print(df_ens.to_string(index=False))
    print()

    # Base paper comparison (from paper: Dash et al. 2025)
    print(f"{'='*65}")
    print("  COMPARISON vs BASE PAPER (Dash et al., Sci. Reports 2025)")
    print(f"  NSL-KDD: SSA-LSTM => Acc=97.8%, F1=97.9%, AUC=98.1%")
    print(f"  (Our ensemble target: exceed on all metrics)")
    print(f"{'='*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate all saved models')
    parser.add_argument('--dataset', choices=['nslkdd', 'cicids', 'both'],
                        default='both')
    args = parser.parse_args()

    datasets = ['nslkdd', 'cicids'] if args.dataset == 'both' else [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            results = evaluate_dataset(ds)
            all_results[ds] = results
            print(f"\n[Done] {ds.upper()} evaluation complete."
                  f" Outputs saved -> {os.path.join(OUTPUTS_DIR, ds)}")
        except FileNotFoundError as e:
            print(f"\n[SKIP] {ds.upper()}: {e}")

    if all_results:
        print_final_summary(all_results)


if __name__ == '__main__':
    main()
