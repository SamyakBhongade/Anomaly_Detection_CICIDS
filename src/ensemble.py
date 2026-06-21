"""
ensemble.py
Attention-weighted ensemble fusion + XGBoost stacking meta-classifier.
The four model probability scores are fused via learned attention weights,
then stacked through XGBoost which learns which models to trust per pattern.
This is the core novelty vs the base paper.
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb


# ── Attention-weighted fusion ─────────────────────────────────────────────────

class AttentionWeightedFusion:
    """
    Learns a weight per model using validation performance.
    Weight_i = softmax(AUC_i) so better-performing models get more say.
    """

    def __init__(self):
        self.weights = None       # shape: (n_models,)
        self.model_names = None

    def fit(self, proba_matrix: np.ndarray, y_true: np.ndarray,
            model_names: list = None):
        """
        proba_matrix: (N, n_models) — each column is one model's probabilities
        y_true      : (N,) binary labels
        """
        from sklearn.metrics import roc_auc_score
        n_models = proba_matrix.shape[1]
        self.model_names = model_names or [f'model_{i}' for i in range(n_models)]

        aucs = []
        for i in range(n_models):
            try:
                auc = roc_auc_score(y_true, proba_matrix[:, i])
            except Exception:
                auc = 0.5
            aucs.append(auc)
            print(f"  [{self.model_names[i]}] Val AUC: {auc:.4f}")

        aucs = np.array(aucs)
        # Softmax over AUC scores → attention weights
        exp_aucs = np.exp(aucs - aucs.max())
        self.weights = exp_aucs / exp_aucs.sum()
        print(f"\n[Fusion] Attention weights: "
              + " | ".join(f"{n}={w:.3f}"
                for n, w in zip(self.model_names, self.weights)))
        return self

    def transform(self, proba_matrix: np.ndarray) -> np.ndarray:
        """
        Returns weighted sum of model probabilities (N,).
        Also returns full weighted feature matrix for meta-classifier.
        """
        weighted_proba = (proba_matrix * self.weights).sum(axis=1)
        return weighted_proba

    def save(self, path: str):
        joblib.dump({'weights': self.weights, 'names': self.model_names}, path)
        print(f"[Fusion] Weights saved -> {path}")

    def load(self, path: str):
        d = joblib.load(path)
        self.weights     = d['weights']
        self.model_names = d['names']
        return self


# ── XGBoost meta-classifier ───────────────────────────────────────────────────

class EnsembleMetaClassifier:
    """
    Stacking ensemble:
      Input  : (N, n_models+1) — raw proba scores + weighted fusion score
      Output : binary 0/1 prediction
    XGBoost as meta-learner learns non-linear combinations of model outputs.
    """

    def __init__(self, n_estimators: int = 300,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42):
        self.clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            tree_method='hist',     # fast on CPU and GPU
            device='cuda' if _gpu_available() else 'cpu'
        )
        self.fusion = AttentionWeightedFusion()
        self.threshold = 0.5

    def _build_meta_features(self, proba_matrix: np.ndarray) -> np.ndarray:
        """
        Stack: [raw proba scores | weighted fusion score].
        Shape: (N, n_models + 1)
        """
        weighted = self.fusion.transform(proba_matrix).reshape(-1, 1)
        return np.hstack([proba_matrix, weighted])

    def fit(self, proba_matrix: np.ndarray, y_true: np.ndarray,
            model_names: list = None):
        """
        proba_matrix: (N, n_models) from validation set
        y_true      : (N,) binary labels
        """
        print("\n[Ensemble] Fitting attention fusion weights...")
        self.fusion.fit(proba_matrix, y_true, model_names)

        print("[Ensemble] Building meta-features...")
        X_meta = self._build_meta_features(proba_matrix)

        print(f"[Ensemble] Training XGBoost meta-classifier on shape {X_meta.shape}...")
        self.clf.fit(X_meta, y_true,
                     eval_set=[(X_meta, y_true)],
                     verbose=False)

        # Find optimal threshold via validation F1
        self.threshold = self._find_threshold(proba_matrix, y_true)
        print(f"[Ensemble] Optimal threshold: {self.threshold:.3f}")
        return self

    def _find_threshold(self, proba_matrix, y_true):
        """Grid-search threshold on validation set for best F1."""
        from sklearn.metrics import f1_score
        X_meta = self._build_meta_features(proba_matrix)
        proba  = self.clf.predict_proba(X_meta)[:, 1]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.02):
            preds = (proba >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        print(f"[Ensemble] Best threshold={best_t:.2f} (F1={best_f1:.4f})")
        return best_t

    def predict(self, proba_matrix: np.ndarray) -> np.ndarray:
        X_meta = self._build_meta_features(proba_matrix)
        proba  = self.clf.predict_proba(X_meta)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, proba_matrix: np.ndarray) -> np.ndarray:
        X_meta = self._build_meta_features(proba_matrix)
        return self.clf.predict_proba(X_meta)[:, 1]

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.clf, os.path.join(save_dir, 'meta_xgb.pkl'))
        self.fusion.save(os.path.join(save_dir, 'fusion_weights.pkl'))
        joblib.dump({'threshold': self.threshold},
                    os.path.join(save_dir, 'threshold.pkl'))
        print(f"[Ensemble] Saved -> {save_dir}")

    def load(self, save_dir: str):
        self.clf       = joblib.load(os.path.join(save_dir, 'meta_xgb.pkl'))
        self.fusion.load(os.path.join(save_dir, 'fusion_weights.pkl'))
        self.threshold = joblib.load(
            os.path.join(save_dir, 'threshold.pkl'))['threshold']
        print(f"[Ensemble] Loaded from {save_dir}")
        return self


# ── Helper to collect all model probabilities ─────────────────────────────────

def collect_probabilities(bilstm_model, cnn_model,
                          transformer_model,
                          vae_encoder, vae_decoder,
                          X: np.ndarray) -> np.ndarray:
    """
    Run all 4 models and stack their probability scores.
    Returns proba_matrix of shape (N, 4).
    Columns: [BiLSTM, CNN, Transformer, VAE]
    """
    from bilstm_model     import predict_proba_bilstm
    from cnn_model        import predict_proba_cnn
    from transformer_model import predict_proba_transformer
    from vae_model        import predict_proba_vae

    print("[Ensemble] Collecting model probabilities...")
    p_bilstm = predict_proba_bilstm(bilstm_model, X)
    print(f"  BiLSTM     done — shape: {p_bilstm.shape}")

    p_cnn = predict_proba_cnn(cnn_model, X)
    print(f"  CNN        done — shape: {p_cnn.shape}")

    p_transformer = predict_proba_transformer(transformer_model, X)
    print(f"  Transformer done — shape: {p_transformer.shape}")

    p_vae = predict_proba_vae(vae_encoder, vae_decoder, X)
    print(f"  VAE        done — shape: {p_vae.shape}")

    proba_matrix = np.column_stack([p_bilstm, p_cnn, p_transformer, p_vae])
    print(f"[Ensemble] Proba matrix shape: {proba_matrix.shape}")
    return proba_matrix


def _gpu_available() -> bool:
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except Exception:
        return False
