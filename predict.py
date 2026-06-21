"""
predict.py
==========
Inference module for the Cognitive IDS Anomaly Detection System.
Loads all trained models from inference_models/ and predicts on new data.

Usage (from Python):
    from predict import AnomalyDetector
    detector = AnomalyDetector()
    result = detector.predict(X)   # X = numpy array of shape (n_samples, 70)

Usage (from command line):
    python predict.py --input sample.csv
"""

import os
import sys
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # silence TF logs

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR  = os.path.join(PROJECT_ROOT, 'inference_models')
SRC_DIR        = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)


class AnomalyDetector:
    """
    End-to-end anomaly detector using the trained 4-model ensemble.

    Pipeline:
        raw features (70-dim) → scaler → 4 models → ensemble → 0/1
    """

    def __init__(self, inference_dir: str = INFERENCE_DIR):
        self.inference_dir = inference_dir
        self._models_loaded = False
        self._load_all()

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_all(self):
        import tensorflow as tf
        # Import custom objects needed for deserialization
        sys.path.insert(0, SRC_DIR)
        from transformer_model import TransformerBlock
        from vae_model import VAELoss, sampling
        from ensemble import EnsembleMetaClassifier

        d = self.inference_dir
        print("[AnomalyDetector] Loading models...")

        # Scaler
        self.scaler = joblib.load(os.path.join(d, 'cicids_scaler.pkl'))

        # Base models
        self.bilstm = tf.keras.models.load_model(
            os.path.join(d, 'bilstm_cicids.keras'))

        self.cnn = tf.keras.models.load_model(
            os.path.join(d, 'cnn_cicids.keras'))

        self.transformer = tf.keras.models.load_model(
            os.path.join(d, 'transformer_cicids.keras'),
            custom_objects={'TransformerBlock': TransformerBlock})

        custom_vae = {'VAELoss': VAELoss, 'sampling': sampling}
        self.encoder = tf.keras.models.load_model(
            os.path.join(d, 'vae', 'vae_encoder.keras'),
            custom_objects=custom_vae)
        self.decoder = tf.keras.models.load_model(
            os.path.join(d, 'vae', 'vae_decoder.keras'),
            custom_objects=custom_vae)

        # Ensemble
        self.ensemble = EnsembleMetaClassifier()
        self.ensemble.load(os.path.join(d, 'ensemble'))

        self._models_loaded = True
        print("[AnomalyDetector] All models loaded OK")

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using the saved MinMaxScaler."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.scaler.transform(X).astype(np.float32)

    # ── Individual model probabilities ─────────────────────────────────────────

    def _get_model_probas(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Returns a (n_samples, 4) matrix of anomaly probabilities
        from [BiLSTM, CNN, Transformer, VAE].
        Uses the original predict_proba_* functions for correctness.
        """
        from bilstm_model       import predict_proba_bilstm
        from cnn_model          import predict_proba_cnn
        from transformer_model  import predict_proba_transformer
        from vae_model          import predict_proba_vae

        p_bilstm      = predict_proba_bilstm(self.bilstm, X_scaled)
        p_cnn         = predict_proba_cnn(self.cnn, X_scaled)
        p_transformer = predict_proba_transformer(self.transformer, X_scaled)
        p_vae         = predict_proba_vae(self.encoder, self.decoder, X_scaled)

        return np.column_stack([p_bilstm, p_cnn, p_transformer, p_vae])

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels for input X.

        Args:
            X: numpy array of shape (n_samples, 70)

        Returns:
            labels: numpy array of int (0 = BENIGN, 1 = ATTACK)
        """
        X_scaled  = self._preprocess(X)
        proba_mat = self._get_model_probas(X_scaled)
        labels    = self.ensemble.predict(proba_mat)
        return labels

    def predict_proba(self, X: np.ndarray) -> dict:
        """
        Returns full probability breakdown from all models + ensemble.

        Returns a dict with keys:
            'bilstm', 'cnn', 'transformer', 'vae', 'ensemble', 'label'
        """
        X_scaled  = self._preprocess(X)
        proba_mat = self._get_model_probas(X_scaled)
        p_ens     = self.ensemble.predict_proba(proba_mat)
        labels    = self.ensemble.predict(proba_mat)

        return {
            'bilstm':      proba_mat[:, 0],
            'cnn':         proba_mat[:, 1],
            'transformer': proba_mat[:, 2],
            'vae':         proba_mat[:, 3],
            'ensemble':    p_ens,
            'label':       labels,             # 0=BENIGN, 1=ATTACK
        }

    def predict_single(self, x: np.ndarray) -> dict:
        """
        Predict a single network flow sample.

        Args:
            x: 1-D numpy array of 70 features

        Returns:
            dict with prediction result and confidence details
        """
        result = self.predict_proba(x.reshape(1, -1))
        label  = result['label'][0]
        conf   = result['ensemble'][0]

        return {
            'prediction':    'ATTACK' if label == 1 else 'BENIGN',
            'label':         int(label),
            'confidence':    round(float(conf), 4),
            'model_scores': {
                'BiLSTM':      round(float(result['bilstm'][0]), 4),
                'CNN':         round(float(result['cnn'][0]), 4),
                'Transformer': round(float(result['transformer'][0]), 4),
                'VAE':         round(float(result['vae'][0]), 4),
                'Ensemble':    round(float(conf), 4),
            }
        }


# ── CLI usage ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Cognitive IDS — Anomaly Detector')
    parser.add_argument('--input', required=True,
                        help='Path to input CSV file (70 feature columns, no label)')
    parser.add_argument('--output', default='predictions.csv',
                        help='Path to save predictions CSV (default: predictions.csv)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    print(f"[CLI] Loading input: {args.input}")
    df = pd.read_csv(args.input)
    X  = df.values.astype(np.float32)
    print(f"[CLI] Input shape: {X.shape}")

    detector = AnomalyDetector()
    results  = detector.predict_proba(X)

    out_df = pd.DataFrame({
        'Prediction':    ['ATTACK' if l == 1 else 'BENIGN' for l in results['label']],
        'Label':         results['label'],
        'Ensemble_Prob': results['ensemble'],
        'BiLSTM_Prob':   results['bilstm'],
        'CNN_Prob':       results['cnn'],
        'Transformer_Prob': results['transformer'],
        'VAE_Score':     results['vae'],
    })

    out_df.to_csv(args.output, index=False)
    print(f"\n[CLI] Results saved -> {args.output}")

    # Summary
    n_attack = (results['label'] == 1).sum()
    n_benign = (results['label'] == 0).sum()
    print(f"\n{'='*40}")
    print(f"  Total samples : {len(results['label'])}")
    print(f"  BENIGN        : {n_benign}")
    print(f"  ATTACK        : {n_attack}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
