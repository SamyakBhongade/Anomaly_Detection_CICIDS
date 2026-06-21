"""
send_traffic.py
===============
Sends real CICIDS2017 network traffic to the detection API.
Run this while the dashboard is open to see live detections.

Usage:
    python send_traffic.py              # sends 20 mixed samples
    python send_traffic.py --n 100      # send 100 samples
    python send_traffic.py --only attack  # send only attack traffic
    python send_traffic.py --only benign  # send only benign traffic
    python send_traffic.py --loop       # keep sending every 2 seconds
"""

import argparse, time, json
import numpy as np
import urllib.request

API_URL   = 'http://localhost:5000/api/detect'
CACHE     = 'data/processed/cicids/cicids_arrays.npz'


def send(X: np.ndarray, attack_types: list = None, ip_addresses: list = None):
    payload_dict = {'data': X.tolist()}
    if attack_types:
        payload_dict['attack_types'] = attack_types
    if ip_addresses:
        payload_dict['ip_addresses'] = ip_addresses
    payload = json.dumps(payload_dict).encode('utf-8')
    req     = urllib.request.Request(
        API_URL,
        data    = payload,
        headers = {'Content-Type': 'application/json'},
        method  = 'POST'
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',    type=int, default=20,  help='Number of samples per batch')
    parser.add_argument('--only', choices=['attack','benign','mixed'], default='mixed')
    parser.add_argument('--loop', action='store_true', default=True, help='Keep sending continuously')
    parser.add_argument('--interval', type=float, default=20.0)
    args = parser.parse_args()
    
    # Force loop and 20s interval as requested
    args.loop = True
    args.interval = 20.0
    args.n = 20

    print(f"[Traffic] Loading CICIDS2017 test data...")
    data   = np.load(CACHE)
    X_test = data['X_test'].astype('float32')
    y_test = data['y_test'].astype(int)

    # The cached arrays are ALREADY SCALED (MinMaxScaler was applied during preprocessing).
    # The /api/detect endpoint expects RAW traffic and applies the scaler internally.
    # So we inverse_transform here to get raw values → API scales them correctly.
    import joblib, os
    scaler_path = os.path.join(os.path.dirname(CACHE), 'cicids_scaler.pkl')
    scaler      = joblib.load(scaler_path)
    X_test      = scaler.inverse_transform(X_test).astype('float32')
    print(f"[Traffic] Inverse-transformed cached data back to raw feature space ✅")

    rng = np.random.default_rng()

    idx_attack = np.where(y_test == 1)[0]
    idx_benign = np.where(y_test == 0)[0]

    if args.only == 'attack':
        label = 'ATTACK-ONLY'
    elif args.only == 'benign':
        label = 'BENIGN-ONLY'
    else:
        label = 'CUSTOM-MIX (15 Benign, 5 Attack)'

    print(f"[Traffic] Pool: {len(X_test):,} samples  |  Mode: {label}")
    print(f"[Traffic] Sending {args.n} samples per batch → {API_URL}\n")

    batch = 1
    while True:
        if args.only == 'attack':
            chosen = rng.choice(idx_attack, size=args.n, replace=False)
        elif args.only == 'benign':
            chosen = rng.choice(idx_benign, size=args.n, replace=False)
        else:
            # Exactly 15 benign and 5 attack
            chosen_b = rng.choice(idx_benign, size=15, replace=False)
            chosen_a = rng.choice(idx_attack, size=5, replace=False)
            chosen = np.concatenate([chosen_b, chosen_a])
            rng.shuffle(chosen)

        X_batch = X_test[chosen]
        y_batch = y_test[chosen]

        # Simulate diverse attack types for visualization
        attack_categories = [
            'DoS', 'DDoS', 'PortScan', 'BruteForce', 'WebAttack', 'Botnet', 
            'Infiltration', 'SQL Injection', 'XSS', 'Command Injection', 'Ransomware'
        ]
        attack_probs = [
            0.25, 0.20, 0.15, 0.10, 0.05, 0.05, 
            0.05, 0.05, 0.04, 0.03, 0.03
        ]
        
        batch_types = []
        for y in y_batch:
            if y == 1:
                batch_types.append(rng.choice(attack_categories, p=attack_probs))
            else:
                batch_types.append('Benign')

        print(f"\n▶ Starting Batch {batch} ({args.n} samples: 15 Benign, 5 Attack)")
        attacks_total = 0
        benign_total = 0

        for i in range(args.n):
            X_single = X_batch[i:i+1]  # Keep 2D array shape for the API
            type_single = [batch_types[i]]

            # Generate realistic fake IP
            if batch_types[i] == 'Benign':
                ip_single = [f"192.168.1.{rng.integers(2, 254)}"]
            else:
                ip_single = [f"{rng.integers(1, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"]

            try:
                resp = send(X_single, attack_types=type_single, ip_addresses=ip_single)
                attacks = resp.get('attacks', 0)
                benign  = resp.get('benign', 0)
                attacks_total += attacks
                benign_total += benign

                status = f"{'🚨 ATTACK' if attacks > 0 else '✅ Benign'}"
                # Add padding for nice alignment
                type_str = batch_types[i][:15].ljust(15) 
                print(f"  Req {i+1:02d}/{args.n} | {status} | Type: {type_str}")

            except Exception as e:
                print(f"  [ERROR] Req {i+1:02d}: {e}")
                print(f"  Make sure the dashboard is running:  python dashboard_app/app.py")

            # Wait 10 seconds between each request, except after the very last one
            if i < args.n - 1:
                time.sleep(10)

        print(f"⏹ Batch {batch} Complete | Total Attack: {attacks_total} | Total Benign: {benign_total}")

        batch += 1
        if not args.loop:
            break
        # Once the batch is complete, the next batch starts immediately (no extra interval needed)

    print("\n[Done] Check your dashboard at http://localhost:5000")


if __name__ == '__main__':
    main()
