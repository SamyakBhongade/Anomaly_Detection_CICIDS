"""
simulator.py
============
Background thread that simulates live network traffic by feeding
real CICIDS2017 test samples through the trained model continuously.
Results are stored in SQLite for the dashboard to display.
"""

import os, sys, time, threading
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from db import insert_batch, init_db

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 10      # samples processed per tick
TICK_INTERVAL   = 0.5     # seconds between batches (20 samples/sec)
CACHE_PATH      = os.path.join(PROJECT_ROOT,
                               'data', 'processed', 'cicids', 'cicids_arrays.npz')


class TrafficSimulator:
    def __init__(self):
        self._running  = False
        self._thread   = None
        self._speed    = TICK_INTERVAL    # adjustable from dashboard
        self.detector  = None
        self._X        = None
        self._y        = None
        self._idx      = 0
        self._lock     = threading.Lock()
        self._throughput = 0.0    # samples/sec (live)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(self):
        """Load model and data (called once at startup)."""
        print("[Simulator] Loading CICIDS2017 test data...")
        data     = np.load(CACHE_PATH)
        self._X  = data['X_test'].astype('float32')
        self._y  = data['y_test'].astype(int)

        # Shuffle once so data isn't time-ordered
        rng = np.random.default_rng(42)
        perm      = rng.permutation(len(self._X))
        self._X   = self._X[perm]
        self._y   = self._y[perm]
        print(f"[Simulator] Loaded {len(self._X):,} samples  "
              f"(benign={( self._y==0).sum():,}  attack={(self._y==1).sum():,})")

        print("[Simulator] Loading detection models...")
        from predict import AnomalyDetector
        self.detector = AnomalyDetector()
        print("[Simulator] Ready ✅")

    # ── Run loop ──────────────────────────────────────────────────────────────

    def _run(self):
        n = len(self._X)
        while self._running:
            t_start = time.time()

            with self._lock:
                speed = self._speed

            # Grab next batch (wraps around when exhausted)
            end   = min(self._idx + BATCH_SIZE, n)
            X_b   = self._X[self._idx:end]
            self._idx = end % n   # wrap

            # Run model inference
            proba_mat   = self.detector._get_model_probas(X_b)
            p_ens       = self.detector.ensemble.predict_proba(proba_mat)
            labels      = self.detector.ensemble.predict(proba_mat)

            # Build DB records
            from datetime import datetime
            ts      = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            records = [
                (ts, int(labels[i]), round(float(p_ens[i]), 5),
                 round(float(proba_mat[i, 0]), 5),
                 round(float(proba_mat[i, 1]), 5),
                 round(float(proba_mat[i, 2]), 5),
                 round(float(proba_mat[i, 3]), 5))
                for i in range(len(labels))
            ]
            insert_batch(records)

            # Throughput tracking
            elapsed = time.time() - t_start
            self._throughput = round(len(labels) / max(elapsed, 0.001), 1)

            # Sleep remainder of tick
            sleep_t = max(0, speed - elapsed)
            time.sleep(sleep_t)

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[Simulator] Started")

    def stop(self):
        self._running = False
        print("[Simulator] Stopped")

    def set_speed(self, interval: float):
        """Change tick interval (lower = faster)."""
        with self._lock:
            self._speed = max(0.1, float(interval))

    def get_throughput(self) -> float:
        return self._throughput

    def is_running(self) -> bool:
        return self._running


# Singleton
_simulator = None

def get_simulator() -> TrafficSimulator:
    global _simulator
    if _simulator is None:
        init_db()
        _simulator = TrafficSimulator()
        _simulator.load()
    return _simulator
