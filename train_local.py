"""
train_local.py
==============
Local training script for the Cognitive IDS ensemble.
Supports checkpoint-based resumable training -- pause anytime, resume later.

Usage:
    python train_local.py --dataset cicids
    python train_local.py --dataset nslkdd
    python train_local.py --dataset both
    python train_local.py --dataset cicids --skip-preprocess
    python train_local.py --dataset cicids --reset   # wipe state, start fresh

How checkpointing works:
    checkpoints/training_state.json tracks which models are done and how many
    epochs were completed. Re-run the same command to resume automatically.
"""

import os, sys, json, time, argparse
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

NSL_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'NSLKDD', 'KDDTrain+.txt')
NSL_TEST_PATH  = os.path.join(PROJECT_ROOT, 'data', 'NSLKDD', 'KDDTest+.txt')
CICIDS_DIR     = os.path.join(PROJECT_ROOT, 'data', 'CICIDS2017')
SAVE_DIR_NSL   = os.path.join(PROJECT_ROOT, 'data', 'processed', 'nslkdd')
SAVE_DIR_CIC   = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cicids')
MODELS_DIR     = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR    = os.path.join(PROJECT_ROOT, 'outputs')
CKPT_DIR       = os.path.join(PROJECT_ROOT, 'checkpoints')

# ── default hyperparameters ────────────────────────────────────────────────────
DEFAULT_BILSTM_PARAMS = {'lstm_units': 128, 'dense_units': 64,
                         'dropout_rate': 0.3, 'l2_reg': 1e-4, 'batch_size': 256}
DEFAULT_CNN_PARAMS    = {'filters': 64, 'dense_units': 64,
                         'dropout_rate': 0.3, 'l2_reg': 1e-4, 'batch_size': 256}
DEFAULT_TF_PARAMS     = {'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
                         'num_blocks': 2, 'dropout_rate': 0.2, 'batch_size': 256}
DEFAULT_VAE_PARAMS    = {'latent_dim': 16, 'hidden_dim': 64,
                         'dropout': 0.2, 'batch_size': 256}
EPOCHS = 100


# ==============================================================================
# Checkpoint state manager
# ==============================================================================

class CheckpointState:
    PENDING = 'pending'
    DONE    = 'done'

    def __init__(self, state_path):
        self.path  = state_path
        self.state = {}
        if os.path.exists(state_path):
            with open(state_path) as f:
                self.state = json.load(f)
            print(f"[Checkpoint] Loaded state from {state_path}")
            self._print_summary()
        else:
            print(f"[Checkpoint] Starting fresh -- state will be saved to {state_path}")

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _key(self, dataset, model):
        return f"{dataset}_{model}"

    def _print_summary(self):
        if not self.state:
            return
        print("\n  +-- Checkpoint summary --------------------------")
        for key, info in self.state.items():
            status = info.get('status', '?')
            ep     = info.get('epochs_done', 0)
            icon   = 'DONE' if status == self.DONE else 'pending'
            print(f"  |  [{icon}]  {key:<30}  ep={ep}")
        print("  +------------------------------------------------\n")

    def is_done(self, dataset, model):
        return self.state.get(self._key(dataset, model), {}).get('status') == self.DONE

    def get_epochs_done(self, dataset, model):
        return self.state.get(self._key(dataset, model), {}).get('epochs_done', 0)

    def mark_started(self, dataset, model, save_path, ckpt_path):
        key = self._key(dataset, model)
        self.state.setdefault(key, {})
        self.state[key].update({'status': self.PENDING,
                                'save_path': save_path,
                                'best_ckpt_path': ckpt_path})
        self._save()

    def update_epochs(self, dataset, model, n):
        key = self._key(dataset, model)
        self.state.setdefault(key, {})
        self.state[key]['epochs_done'] = n
        self.state[key]['updated_at']  = time.strftime('%Y-%m-%dT%H:%M:%S')
        self._save()

    def mark_done(self, dataset, model, n, save_path):
        key = self._key(dataset, model)
        self.state[key].update({'status': self.DONE, 'epochs_done': n,
                                'save_path': save_path,
                                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
        self._save()
        print(f"[Checkpoint] DONE: {key}  ({n} epochs)")


# ==============================================================================
# Epoch counter callback
# ==============================================================================

import tensorflow as tf

class _EpochCounter(tf.keras.callbacks.Callback):
    def __init__(self, state, dataset, model_name, initial_epoch=0):
        super().__init__()
        self.state         = state
        self.dataset       = dataset
        self.model_name    = model_name
        self.initial_epoch = initial_epoch

    def on_epoch_end(self, epoch, logs=None):
        self.state.update_epochs(self.dataset, self.model_name,
                                 self.initial_epoch + epoch + 1)


# ==============================================================================
# Shared callbacks
# ==============================================================================

def _build_callbacks(ckpt_path, state, dataset, model_name, initial_epoch,
                     patience_es=10, patience_lr=5):
    from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                            ModelCheckpoint)
    os.makedirs(CKPT_DIR, exist_ok=True)
    return [
        ModelCheckpoint(filepath=ckpt_path, monitor='val_loss',
                        save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=patience_es,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=patience_lr, min_lr=1e-6, verbose=1),
        _EpochCounter(state, dataset, model_name, initial_epoch),
    ]


def _ckpt_path(dataset, model_name):
    return os.path.join(CKPT_DIR, f'{dataset}_{model_name}_best.weights.h5')


# ==============================================================================
# Resumable model trainers
# ==============================================================================

def train_bilstm_resumable(X_tr, y_tr, X_vl, y_vl, dataset, state, params, epochs):
    from bilstm_model import build_bilstm
    model_name = 'bilstm'
    save_path  = os.path.join(MODELS_DIR, f'bilstm_{dataset}.keras')
    ckpt       = _ckpt_path(dataset, model_name)

    if state.is_done(dataset, model_name):
        print(f"[BiLSTM/{dataset}] Already done -- loading.")
        return tf.keras.models.load_model(save_path), None

    dim           = X_tr.shape[1]
    initial_epoch = state.get_epochs_done(dataset, model_name)
    X_tr_r = X_tr.reshape(-1, dim, 1)
    X_vl_r = X_vl.reshape(-1, dim, 1)

    model = build_bilstm(dim, params['lstm_units'], params['dense_units'],
                         params['dropout_rate'], params['l2_reg'])

    if initial_epoch > 0 and os.path.exists(ckpt + '.index'):
        print(f"[BiLSTM/{dataset}] Resuming from epoch {initial_epoch}.")
        model.load_weights(ckpt)
    elif initial_epoch > 0 and os.path.exists(save_path):
        model = tf.keras.models.load_model(save_path)
    else:
        model.summary()

    state.mark_started(dataset, model_name, save_path, ckpt)
    hist = model.fit(X_tr_r, y_tr, validation_data=(X_vl_r, y_vl),
                     epochs=epochs, initial_epoch=initial_epoch,
                     batch_size=params.get('batch_size', 256),
                     callbacks=_build_callbacks(ckpt, state, dataset, model_name, initial_epoch),
                     verbose=1)

    if os.path.exists(ckpt + '.index'):
        model.load_weights(ckpt)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(save_path)
    state.mark_done(dataset, model_name, hist.epoch[-1] + 1, save_path)
    return model, hist


def train_cnn_resumable(X_tr, y_tr, X_vl, y_vl, dataset, state, params, epochs):
    from cnn_model import build_cnn
    model_name = 'cnn'
    save_path  = os.path.join(MODELS_DIR, f'cnn_{dataset}.keras')
    ckpt       = _ckpt_path(dataset, model_name)

    if state.is_done(dataset, model_name):
        print(f"[CNN/{dataset}] Already done -- loading.")
        return tf.keras.models.load_model(save_path), None

    dim           = X_tr.shape[1]
    initial_epoch = state.get_epochs_done(dataset, model_name)
    X_tr_r = X_tr.reshape(-1, dim, 1)
    X_vl_r = X_vl.reshape(-1, dim, 1)

    model = build_cnn(dim, params['filters'], params['dense_units'],
                      params['dropout_rate'], params['l2_reg'])

    if initial_epoch > 0 and os.path.exists(ckpt + '.index'):
        print(f"[CNN/{dataset}] Resuming from epoch {initial_epoch}.")
        model.load_weights(ckpt)
    elif initial_epoch > 0 and os.path.exists(save_path):
        model = tf.keras.models.load_model(save_path)
    else:
        model.summary()

    state.mark_started(dataset, model_name, save_path, ckpt)
    hist = model.fit(X_tr_r, y_tr, validation_data=(X_vl_r, y_vl),
                     epochs=epochs, initial_epoch=initial_epoch,
                     batch_size=params.get('batch_size', 256),
                     callbacks=_build_callbacks(ckpt, state, dataset, model_name, initial_epoch),
                     verbose=1)

    if os.path.exists(ckpt + '.index'):
        model.load_weights(ckpt)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(save_path)
    state.mark_done(dataset, model_name, hist.epoch[-1] + 1, save_path)
    return model, hist


def train_transformer_resumable(X_tr, y_tr, X_vl, y_vl, dataset, state, params, epochs):
    from transformer_model import build_transformer, TransformerBlock
    model_name = 'transformer'
    save_path  = os.path.join(MODELS_DIR, f'transformer_{dataset}.keras')
    ckpt       = _ckpt_path(dataset, model_name)
    custom_obj = {'TransformerBlock': TransformerBlock}

    if state.is_done(dataset, model_name):
        print(f"[Transformer/{dataset}] Already done -- loading.")
        return tf.keras.models.load_model(save_path, custom_objects=custom_obj), None

    dim           = X_tr.shape[1]
    initial_epoch = state.get_epochs_done(dataset, model_name)
    X_tr_r = X_tr.reshape(-1, dim, 1)
    X_vl_r = X_vl.reshape(-1, dim, 1)

    model = build_transformer(dim, params['embed_dim'], params['num_heads'],
                              params['ff_dim'], params['num_blocks'],
                              dropout_rate=params['dropout_rate'])

    if initial_epoch > 0 and os.path.exists(ckpt + '.index'):
        print(f"[Transformer/{dataset}] Resuming from epoch {initial_epoch}.")
        model.load_weights(ckpt)
    elif initial_epoch > 0 and os.path.exists(save_path):
        model = tf.keras.models.load_model(save_path, custom_objects=custom_obj)
    else:
        model.summary()

    state.mark_started(dataset, model_name, save_path, ckpt)
    hist = model.fit(X_tr_r, y_tr, validation_data=(X_vl_r, y_vl),
                     epochs=epochs, initial_epoch=initial_epoch,
                     batch_size=params.get('batch_size', 256),
                     callbacks=_build_callbacks(ckpt, state, dataset, model_name, initial_epoch),
                     verbose=1)

    if os.path.exists(ckpt + '.index'):
        model.load_weights(ckpt)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(save_path)
    state.mark_done(dataset, model_name, hist.epoch[-1] + 1, save_path)
    return model, hist


def train_vae_resumable(X_tr, y_tr, dataset, state, params, epochs):
    from vae_model import build_vae, VAELoss
    model_name = 'vae'
    save_dir   = os.path.join(MODELS_DIR, f'vae_{dataset}')
    ckpt       = _ckpt_path(dataset, model_name)
    enc_path   = os.path.join(save_dir, 'vae_encoder.keras')
    dec_path   = os.path.join(save_dir, 'vae_decoder.keras')
    full_path  = os.path.join(save_dir, 'vae_full.keras')
    custom_obj = {'VAELoss': VAELoss}

    if state.is_done(dataset, model_name):
        print(f"[VAE/{dataset}] Already done -- loading.")
        vae     = tf.keras.models.load_model(full_path, custom_objects=custom_obj)
        encoder = tf.keras.models.load_model(enc_path,  custom_objects=custom_obj)
        decoder = tf.keras.models.load_model(dec_path,  custom_objects=custom_obj)
        return vae, encoder, decoder, None

    normal_mask   = (y_tr == 0)
    X_normal      = X_tr[normal_mask]
    dim           = X_tr.shape[1]
    initial_epoch = state.get_epochs_done(dataset, model_name)
    print(f"[VAE/{dataset}] Training on {len(X_normal):,} normal samples only")

    vae, encoder, decoder = build_vae(dim, params['latent_dim'],
                                      params['hidden_dim'], params['dropout'])

    if initial_epoch > 0 and os.path.exists(ckpt + '.index'):
        print(f"[VAE/{dataset}] Resuming from epoch {initial_epoch}.")
        vae.load_weights(ckpt)
    elif initial_epoch > 0 and os.path.exists(full_path):
        vae     = tf.keras.models.load_model(full_path, custom_objects=custom_obj)
        encoder = tf.keras.models.load_model(enc_path,  custom_objects=custom_obj)
        decoder = tf.keras.models.load_model(dec_path,  custom_objects=custom_obj)
    else:
        vae.summary()

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        ModelCheckpoint(filepath=ckpt, monitor='val_loss',
                        save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        _EpochCounter(state, dataset, model_name, initial_epoch),
    ]

    state.mark_started(dataset, model_name, full_path, ckpt)
    hist = vae.fit(X_normal, X_normal, validation_split=0.1,
                   epochs=epochs, initial_epoch=initial_epoch,
                   batch_size=params.get('batch_size', 256),
                   callbacks=callbacks, verbose=1)

    if os.path.exists(ckpt + '.index'):
        vae.load_weights(ckpt)
    os.makedirs(save_dir, exist_ok=True)
    vae.save(full_path)
    encoder.save(enc_path)
    decoder.save(dec_path)
    state.mark_done(dataset, model_name, hist.epoch[-1] + 1, save_dir)
    return vae, encoder, decoder, hist


# ==============================================================================
# Ensemble + Evaluation
# ==============================================================================

def train_ensemble(bilstm_m, cnn_m, tf_m, enc, dec, X_vl, y_vl, dataset, state):
    from ensemble import collect_probabilities, EnsembleMetaClassifier
    ens_key  = f'{dataset}_ensemble'
    save_dir = os.path.join(MODELS_DIR, f'ensemble_{dataset}')

    if state.state.get(ens_key, {}).get('status') == 'done':
        print(f"[Ensemble/{dataset}] Already trained -- loading.")
        ens = EnsembleMetaClassifier()
        ens.load(save_dir)
        return ens

    print(f"\n===== Training Ensemble ({dataset.upper()}) =====")
    proba_val = collect_probabilities(bilstm_m, cnn_m, tf_m, enc, dec, X_vl)
    ens = EnsembleMetaClassifier()
    ens.fit(proba_val, y_vl, model_names=['BiLSTM', 'CNN', 'Transformer', 'VAE'])
    ens.save(save_dir)
    state.state[ens_key] = {'status': 'done', 'save_dir': save_dir}
    state._save()
    return ens


def run_evaluation(bilstm_m, cnn_m, tf_m, enc, dec, ens, X_test, y_test, dataset):
    from ensemble import collect_probabilities
    from bilstm_model      import predict_proba_bilstm
    from cnn_model         import predict_proba_cnn
    from transformer_model import predict_proba_transformer
    from vae_model         import predict_proba_vae
    from evaluate import (full_evaluation, plot_roc_curves,
                          plot_metrics_bar, save_metrics_table)
    import pandas as pd

    out_dir = os.path.join(OUTPUTS_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== Evaluating ({dataset.upper()}) =====")

    p_b = predict_proba_bilstm(bilstm_m, X_test)
    p_c = predict_proba_cnn(cnn_m, X_test)
    p_t = predict_proba_transformer(tf_m, X_test)
    p_v = predict_proba_vae(enc, dec, X_test)

    proba_test = collect_probabilities(bilstm_m, cnn_m, tf_m, enc, dec, X_test)
    p_ens  = ens.predict_proba(proba_test)
    y_pred = ens.predict(proba_test)

    tag = dataset.upper()
    all_metrics = []
    for name, proba in [('BiLSTM', p_b), ('CNN', p_c),
                        ('Transformer', p_t), ('VAE', p_v)]:
        preds = (proba >= 0.5).astype(int)
        all_metrics.append(full_evaluation(y_test, preds, proba, name, tag, out_dir))

    all_metrics.append(full_evaluation(y_test, y_pred, p_ens, 'Ensemble (Ours)', tag, out_dir))

    plot_roc_curves([{'name': 'BiLSTM', 'proba': p_b},
                     {'name': 'CNN', 'proba': p_c},
                     {'name': 'Transformer', 'proba': p_t},
                     {'name': 'VAE', 'proba': p_v},
                     {'name': 'Ensemble', 'proba': p_ens}],
                    y_test, out_dir, tag)
    plot_metrics_bar(all_metrics, out_dir, tag)
    save_metrics_table(all_metrics, out_dir, tag)

    df = pd.DataFrame(all_metrics)[['Model', 'Accuracy', 'Precision', 'Recall',
                                    'F1-Score', 'AUC-ROC', 'FPR']]
    print(f"\n{'='*60}\nRESULTS -- {tag}\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nOutputs saved -> {out_dir}")
    return all_metrics


# ==============================================================================
# Main pipeline per dataset
# ==============================================================================

def run_dataset(dataset, state, skip_preprocess=False):
    from sklearn.model_selection import train_test_split

    print(f"\n{'#'*60}\n#  DATASET: {dataset.upper()}\n{'#'*60}\n")

    processed_dir = SAVE_DIR_NSL if dataset == 'nslkdd' else SAVE_DIR_CIC
    cache_file    = os.path.join(processed_dir, f'{dataset}_arrays.npz')

    if skip_preprocess and os.path.exists(cache_file):
        print(f"[Preprocess] Loading cached arrays from {cache_file}")
        data = np.load(cache_file)
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        print(f"[Preprocess] Train:{X_train.shape}  Test:{X_test.shape}")
    else:
        if dataset == 'nslkdd':
            from preprocess import prepare_nslkdd
            X_train, X_test, y_train, y_test, _ = prepare_nslkdd(
                train_path=NSL_TRAIN_PATH, test_path=NSL_TEST_PATH,
                save_dir=SAVE_DIR_NSL, use_smote=True, random_state=42)
        else:
            from preprocess import prepare_cicids
            X_train, X_test, y_train, y_test, _ = prepare_cicids(
                data_dir=CICIDS_DIR, save_dir=SAVE_DIR_CIC,
                use_smote=True, random_state=42)
        os.makedirs(processed_dir, exist_ok=True)
        np.savez_compressed(cache_file,
                            X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)
        print(f"[Preprocess] Arrays cached -> {cache_file}")

    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    print(f"[Split] Train:{X_tr.shape}  Val:{X_vl.shape}  Test:{X_test.shape}")

    # Load SSA params or use defaults
    ssa_path = os.path.join(MODELS_DIR, 'ssa_best_params.pkl')
    if os.path.exists(ssa_path):
        import joblib
        best          = joblib.load(ssa_path)
        bilstm_params = best.get('bilstm', DEFAULT_BILSTM_PARAMS)
        cnn_params    = best.get('cnn',    DEFAULT_CNN_PARAMS)
        tf_params     = best.get('transformer', DEFAULT_TF_PARAMS)
        print("[Params] Loaded SSA-tuned hyperparameters.")
    else:
        bilstm_params = DEFAULT_BILSTM_PARAMS
        cnn_params    = DEFAULT_CNN_PARAMS
        tf_params     = DEFAULT_TF_PARAMS
        print("[Params] Using default hyperparameters.")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n===== Training BiLSTM ({dataset.upper()}) =====")
    bilstm_m, _ = train_bilstm_resumable(X_tr, y_tr, X_vl, y_vl, dataset, state, bilstm_params, EPOCHS)

    print(f"\n===== Training CNN ({dataset.upper()}) =====")
    cnn_m, _ = train_cnn_resumable(X_tr, y_tr, X_vl, y_vl, dataset, state, cnn_params, EPOCHS)

    print(f"\n===== Training Transformer ({dataset.upper()}) =====")
    tf_m, _ = train_transformer_resumable(X_tr, y_tr, X_vl, y_vl, dataset, state, tf_params, EPOCHS)

    print(f"\n===== Training VAE ({dataset.upper()}) =====")
    vae_m, enc, dec, _ = train_vae_resumable(X_tr, y_tr, dataset, state, DEFAULT_VAE_PARAMS, EPOCHS)

    ens = train_ensemble(bilstm_m, cnn_m, tf_m, enc, dec, X_vl, y_vl, dataset, state)
    run_evaluation(bilstm_m, cnn_m, tf_m, enc, dec, ens, X_test, y_test, dataset)


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    global EPOCHS

    parser = argparse.ArgumentParser(description='Cognitive IDS local training')
    parser.add_argument('--dataset', choices=['nslkdd', 'cicids', 'both'],
                        default='cicids')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip preprocessing if cached .npz exists')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Max epochs per model (default: {EPOCHS})')
    parser.add_argument('--reset', action='store_true',
                        help='Clear checkpoint state and restart')
    args = parser.parse_args()

    EPOCHS = args.epochs

    for d in [MODELS_DIR, OUTPUTS_DIR, CKPT_DIR]:
        os.makedirs(d, exist_ok=True)

    state_path = os.path.join(CKPT_DIR, 'training_state.json')
    if args.reset and os.path.exists(state_path):
        os.remove(state_path)
        print("[Reset] Checkpoint state cleared.")

    state    = CheckpointState(state_path)
    datasets = ['nslkdd', 'cicids'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets:
        run_dataset(ds, state, skip_preprocess=args.skip_preprocess)

    print(f"\n{'='*60}\nALL TRAINING COMPLETE\n{'='*60}")


if __name__ == '__main__':
    main()
