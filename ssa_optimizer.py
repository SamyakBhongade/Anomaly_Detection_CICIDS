"""
ssa_optimizer.py
Salp Swarm Algorithm (SSA) for hyperparameter optimization.
Tunes BiLSTM, CNN, Transformer, and VAE hyperparameters simultaneously.
SSA is kept from the base paper as the proven best optimizer,
but extended here to tune ALL four models instead of just LSTM.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict


class SalpSwarmOptimizer:
    """
    SSA implementation.
    Salps form a chain: the leader navigates toward food (best solution),
    followers update based on adjacent salp positions.
    """

    def __init__(self,
                 n_salps:   int   = 10,
                 max_iter:  int   = 30,
                 lb:        list  = None,
                 ub:        list  = None,
                 dim:       int   = None,
                 seed:      int   = 42):
        """
        n_salps  : population size
        max_iter : number of iterations
        lb / ub  : lower / upper bounds (lists of length dim)
        dim      : number of hyperparameters to optimize
        """
        self.n_salps  = n_salps
        self.max_iter = max_iter
        self.lb       = np.array(lb) if lb else None
        self.ub       = np.array(ub) if ub else None
        self.dim      = dim or (len(lb) if lb else None)
        self.rng      = np.random.RandomState(seed)
        self.best_pos = None
        self.best_fit = np.inf
        self.convergence = []

    def _initialize(self) -> np.ndarray:
        """Random uniform initialization within bounds."""
        return self.lb + self.rng.rand(self.n_salps, self.dim) * (self.ub - self.lb)

    def _clip(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, self.lb, self.ub)

    def optimize(self, fitness_fn: Callable) -> Tuple[np.ndarray, float, List]:
        """
        Run SSA optimization.
        fitness_fn: function(position_array) -> float (lower is better)
        Returns: best_position, best_fitness, convergence_curve
        """
        positions = self._initialize()
        fitness   = np.array([fitness_fn(positions[i]) for i in range(self.n_salps)])

        best_idx      = np.argmin(fitness)
        self.best_pos = positions[best_idx].copy()
        self.best_fit = fitness[best_idx]
        self.convergence = []

        print(f"\n[SSA] Starting optimization: {self.n_salps} salps, {self.max_iter} iterations")
        print(f"[SSA] Search space dim: {self.dim}")
        print(f"[SSA] Initial best fitness: {self.best_fit:.6f}\n")

        for iteration in range(self.max_iter):
            # Eq. 3.2 — adaptive coefficient c1
            c1 = 2 * np.exp(-((4 * iteration / self.max_iter) ** 2))

            for i in range(self.n_salps):
                if i == 0:
                    # Leader salp: move toward food source
                    c2 = self.rng.rand(self.dim)
                    c3 = self.rng.rand(self.dim)
                    move = c1 * ((self.ub - self.lb) * c2 + self.lb)
                    new_pos = np.where(c3 >= 0.5,
                                       self.best_pos + move,
                                       self.best_pos - move)
                else:
                    # Follower salp: average of adjacent positions
                    new_pos = 0.5 * (positions[i] + positions[i - 1])

                positions[i] = self._clip(new_pos)

            # Evaluate fitness
            fitness = np.array([fitness_fn(positions[i]) for i in range(self.n_salps)])

            # Update global best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fit:
                self.best_pos = positions[best_idx].copy()
                self.best_fit = fitness[best_idx]

            self.convergence.append(self.best_fit)
            print(f"[SSA] Iter {iteration+1:3d}/{self.max_iter} | "
                  f"Best fitness: {self.best_fit:.6f}")

        print(f"\n[SSA] Optimization complete. Best fitness: {self.best_fit:.6f}")
        print(f"[SSA] Best position: {self.best_pos}")
        return self.best_pos, self.best_fit, self.convergence


# ── Search space definitions ──────────────────────────────────────────────────

BILSTM_SPACE = {
    # [lstm_units, dense_units, dropout*100, l2_exp*10, batch_size_idx]
    'lb':  [32,   32,  10,  1, 0],
    'ub':  [256, 128,  50, 40, 3],
    'dim': 5,
    'names': ['lstm_units', 'dense_units', 'dropout*100', 'l2_exp*10', 'batch_idx']
}

CNN_SPACE = {
    # [filters, dense_units, dropout*100, l2_exp*10, batch_size_idx]
    'lb':  [32,   32,  10,  1, 0],
    'ub':  [256, 128,  50, 40, 3],
    'dim': 5,
    'names': ['filters', 'dense_units', 'dropout*100', 'l2_exp*10', 'batch_idx']
}

TRANSFORMER_SPACE = {
    # [embed_dim_idx, num_heads_idx, ff_dim, num_blocks, dropout*100]
    'lb':  [0, 0,   64, 1, 10],
    'ub':  [3, 2,  256, 4, 40],
    'dim': 5,
    'names': ['embed_idx', 'head_idx', 'ff_dim', 'num_blocks', 'dropout*100']
}

VAE_SPACE = {
    # [latent_dim, hidden_dim, dropout*100]
    'lb':  [ 8,  32, 10],
    'ub':  [64, 256, 40],
    'dim': 3,
    'names': ['latent_dim', 'hidden_dim', 'dropout*100']
}

BATCH_SIZES  = [64, 128, 256, 512]
EMBED_DIMS   = [32, 64, 128, 256]
NUM_HEADS    = [2, 4, 8]


# ── Decoder helpers ───────────────────────────────────────────────────────────

def decode_bilstm_params(pos: np.ndarray) -> Dict:
    return {
        'lstm_units':   int(np.clip(round(pos[0]), 32, 256)),
        'dense_units':  int(np.clip(round(pos[1]), 32, 128)),
        'dropout_rate': float(np.clip(pos[2] / 100, 0.1, 0.5)),
        'l2_reg':       float(10 ** (-np.clip(pos[3] / 10, 1, 4))),
        'batch_size':   BATCH_SIZES[int(np.clip(round(pos[4]), 0, 3))],
    }


def decode_cnn_params(pos: np.ndarray) -> Dict:
    return {
        'filters':      int(np.clip(round(pos[0]), 32, 256)),
        'dense_units':  int(np.clip(round(pos[1]), 32, 128)),
        'dropout_rate': float(np.clip(pos[2] / 100, 0.1, 0.5)),
        'l2_reg':       float(10 ** (-np.clip(pos[3] / 10, 1, 4))),
        'batch_size':   BATCH_SIZES[int(np.clip(round(pos[4]), 0, 3))],
    }


def decode_transformer_params(pos: np.ndarray) -> Dict:
    return {
        'embed_dim':    EMBED_DIMS[int(np.clip(round(pos[0]), 0, 3))],
        'num_heads':    NUM_HEADS[int(np.clip(round(pos[1]), 0, 2))],
        'ff_dim':       int(np.clip(round(pos[2]), 64, 256)),
        'num_blocks':   int(np.clip(round(pos[3]), 1, 4)),
        'dropout_rate': float(np.clip(pos[4] / 100, 0.1, 0.5)),
    }


def decode_vae_params(pos: np.ndarray) -> Dict:
    return {
        'latent_dim': int(np.clip(round(pos[0]),  8,  64)),
        'hidden_dim': int(np.clip(round(pos[1]), 32, 256)),
        'dropout':    float(np.clip(pos[2] / 100, 0.1, 0.5)),
    }


# ── Fitness function builders ─────────────────────────────────────────────────

def make_bilstm_fitness(X_tr, y_tr, X_vl, y_vl, epochs=20):
    """Returns a fitness function for BiLSTM hyperparameter search."""
    from bilstm_model import build_bilstm
    import tensorflow as tf

    def fitness(pos):
        p = decode_bilstm_params(pos)
        try:
            model = build_bilstm(
                input_dim=X_tr.shape[1],
                lstm_units=p['lstm_units'],
                dense_units=p['dense_units'],
                dropout_rate=p['dropout_rate'],
                l2_reg=p['l2_reg']
            )
            X_r = X_tr.reshape(-1, X_tr.shape[1], 1)
            X_v = X_vl.reshape(-1, X_vl.shape[1], 1)
            hist = model.fit(X_r, y_tr,
                             validation_data=(X_v, y_vl),
                             epochs=epochs, batch_size=p['batch_size'],
                             verbose=0)
            val_loss = min(hist.history['val_loss'])
            tf.keras.backend.clear_session()
            print(f"  [fitness] params={p} -> val_loss={val_loss:.5f}")
            return val_loss
        except Exception as e:
            print(f"  [fitness] ERROR: {e}")
            return 1.0

    return fitness


def make_cnn_fitness(X_tr, y_tr, X_vl, y_vl, epochs=20):
    from cnn_model import build_cnn
    import tensorflow as tf

    def fitness(pos):
        p = decode_cnn_params(pos)
        try:
            model = build_cnn(
                input_dim=X_tr.shape[1],
                filters=p['filters'],
                dense_units=p['dense_units'],
                dropout_rate=p['dropout_rate'],
                l2_reg=p['l2_reg']
            )
            X_r = X_tr.reshape(-1, X_tr.shape[1], 1)
            X_v = X_vl.reshape(-1, X_vl.shape[1], 1)
            hist = model.fit(X_r, y_tr,
                             validation_data=(X_v, y_vl),
                             epochs=epochs, batch_size=p['batch_size'],
                             verbose=0)
            val_loss = min(hist.history['val_loss'])
            tf.keras.backend.clear_session()
            print(f"  [fitness] params={p} -> val_loss={val_loss:.5f}")
            return val_loss
        except Exception as e:
            print(f"  [fitness] ERROR: {e}")
            return 1.0

    return fitness


def make_transformer_fitness(X_tr, y_tr, X_vl, y_vl, epochs=20):
    from transformer_model import build_transformer
    import tensorflow as tf

    def fitness(pos):
        p = decode_transformer_params(pos)
        # ensure embed_dim divisible by num_heads
        while p['embed_dim'] % p['num_heads'] != 0:
            p['num_heads'] = max(1, p['num_heads'] - 1)
        try:
            model = build_transformer(
                input_dim=X_tr.shape[1],
                embed_dim=p['embed_dim'],
                num_heads=p['num_heads'],
                ff_dim=p['ff_dim'],
                num_blocks=p['num_blocks'],
                dropout_rate=p['dropout_rate']
            )
            X_r = X_tr.reshape(-1, X_tr.shape[1], 1)
            X_v = X_vl.reshape(-1, X_vl.shape[1], 1)
            hist = model.fit(X_r, y_tr,
                             validation_data=(X_v, y_vl),
                             epochs=epochs, batch_size=128,
                             verbose=0)
            val_loss = min(hist.history['val_loss'])
            tf.keras.backend.clear_session()
            print(f"  [fitness] params={p} -> val_loss={val_loss:.5f}")
            return val_loss
        except Exception as e:
            print(f"  [fitness] ERROR: {e}")
            return 1.0

    return fitness
