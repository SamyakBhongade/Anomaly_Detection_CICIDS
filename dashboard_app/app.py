"""
app.py
======
Flask server for the Cognitive IDS Anomaly Detection Dashboard.
Traffic is submitted manually via file upload or API.

Run:  python app.py
Open: http://localhost:5000
"""

import os, sys, json, time
import numpy as np
from flask import Flask, render_template, jsonify, Response, request

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(DASHBOARD_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from db import (get_stats, get_recent_feed, get_latest_id,
                get_new_attacks_since, reset_db, init_db, insert_batch,
                get_ip_history, block_ip, is_ip_blocked)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024   # 200 MB max upload

# ── Load model once at startup ────────────────────────────────────────────────
print("[App] Loading detection models...")
from predict import AnomalyDetector
detector = AnomalyDetector()
print("[App] Models ready OK\n")


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── REST API ──────────────────────────────────────────────────────────────────

@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'ok',
        'ml_models_loaded': detector is not None,
        'database': 'sqlite',
    })


@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())


@app.route('/api/feed')
def api_feed():
    return jsonify(get_recent_feed(50))


@app.route('/api/reset', methods=['POST'])
def api_reset():
    reset_db()
    return jsonify({'ok': True})


@app.route('/api/block_ip', methods=['POST'])
def api_block_ip():
    data = request.get_json()
    if not data or 'ip' not in data:
        return jsonify({'error': 'No IP provided'}), 400
    block_ip(data['ip'])
    return jsonify({'ok': True, 'ip': data['ip']})


@app.route('/api/ip_history')
def api_ip_history():
    ip = request.args.get('ip')
    if not ip:
        return jsonify({'error': 'No IP provided'}), 400
    history = get_ip_history(ip)
    return jsonify({'ip': ip, 'history': history, 'is_blocked': is_ip_blocked(ip)})


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    Accept traffic data and run it through the model.

    Two modes:
    1. File upload  → multipart/form-data  key='file'  (CSV, no header or with header)
    2. JSON payload → application/json     { "data": [[f1,f2,...], ...] }

    Returns:
        { total, attacks, benign, results: [{label, confidence, ...}, ...] }
    """
    import pandas as pd
    from datetime import datetime

    try:
        # ── Parse input ──────────────────────────────────────────────────────
        attack_types_in = []   # optional per-sample attack type labels
        ip_addresses_in = []   # optional per-sample ip addresses

        if request.is_json:
            payload         = request.get_json()
            X               = np.array(payload['data'], dtype='float32')
            attack_types_in = payload.get('attack_types', [])  # optional
            ip_addresses_in = payload.get('ip_addresses', [])

        elif 'file' in request.files:
            f  = request.files['file']
            df = pd.read_csv(f)
            for col in ['Label', 'label', 'class', 'Class']:
                if col in df.columns:
                    df = df.drop(columns=[col])
            X = df.values.astype('float32')

        else:
            return jsonify({'error': 'Send JSON payload or CSV file'}), 400

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = len(X)

        # ── Run model ─────────────────────────────────────────────────────────
        X_scaled  = detector._preprocess(X)
        proba_mat = detector._get_model_probas(X_scaled)
        p_ens     = detector.ensemble.predict_proba(proba_mat)
        labels    = detector.ensemble.predict(proba_mat)

        # ── Resolve attack type per sample ────────────────────────────────────
        def resolve_type(i, label):
            if label == 0:
                return 'Benign'
            if i < len(attack_types_in) and attack_types_in[i]:
                return str(attack_types_in[i])
            return 'Unknown Attack'

        def resolve_ip(i):
            if i < len(ip_addresses_in) and ip_addresses_in[i]:
                return str(ip_addresses_in[i])
            return f"192.168.1.{np.random.randint(2, 254)}"

        # ── Check for blocked IPs ─────────────────────────────────────────────
        for i in range(n_samples):
            ip = resolve_ip(i)
            if is_ip_blocked(ip):
                return jsonify({'error': 'Forbidden. IP is blocked.', 'ip': ip}), 403

        # ── Save to DB ────────────────────────────────────────────────────────
        ts      = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        records = [
            (ts, resolve_ip(i), int(labels[i]),
             resolve_type(i, labels[i]),
             round(float(p_ens[i]), 5),
             round(float(proba_mat[i, 0]), 5),
             round(float(proba_mat[i, 1]), 5),
             round(float(proba_mat[i, 2]), 5),
             round(float(proba_mat[i, 3]), 5))
            for i in range(n_samples)
        ]
        insert_batch(records)

        n_attacks = int((labels == 1).sum())
        n_benign  = n_samples - n_attacks

        # ── Build per-row results (max 200 rows returned to UI) ───────────────
        results = [
            {
                'label':       int(labels[i]),
                'confidence':  round(float(p_ens[i]), 4),
                'bilstm':      round(float(proba_mat[i, 0]), 4),
                'cnn':         round(float(proba_mat[i, 1]), 4),
                'transformer': round(float(proba_mat[i, 2]), 4),
                'vae':         round(float(proba_mat[i, 3]), 4),
            }
            for i in range(min(n_samples, 200))
        ]

        return jsonify({
            'total':   n_samples,
            'attacks': n_attacks,
            'benign':  n_benign,
            'results': results,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── SSE stream ────────────────────────────────────────────────────────────────

@app.route('/api/stream')
def api_stream():
    """Push stats + attack alerts to frontend every second."""
    def generate():
        last_id = get_latest_id()
        while True:
            time.sleep(1)
            stats  = get_stats()
            alerts = get_new_attacks_since(last_id)
            if alerts:
                last_id = alerts[-1]['id']
            yield f"data: {json.dumps({'stats': stats, 'alerts': alerts})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    print("=" * 55)
    print("  Cognitive IDS Dashboard")
    print("  Open your browser -> http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, threaded=True, port=5000)
