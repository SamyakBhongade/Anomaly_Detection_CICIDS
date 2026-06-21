# Local Training Guide — Cognitive IDS

Train the 4-model ensemble on your local machine with **full pause/resume support**.
You can stop at any time (Ctrl+C) and pick up exactly where you left off.

---

## 1. Prerequisites

```bash
pip install tensorflow scikit-learn imbalanced-learn xgboost seaborn joblib numpy pandas
```

> **No GPU required**, but a GPU (CUDA) will make training much faster.
> If you have an NVIDIA GPU, TensorFlow will auto-detect it.

---

## 2. Dataset Setup

Place your data files like this:

```
cognitive_ids/
├── data/
│   ├── NSLKDD/
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   └── CICIDS2017/
│       ├── Monday-WorkingHours.pcap_ISCX.csv
│       ├── Tuesday-WorkingHours.pcap_ISCX.csv
│       ├── Wednesday-workingHours.pcap_ISCX.csv
│       ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│       ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│       ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│       ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│       └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

---

## 3. Running Training

Open a terminal in the `cognitive_ids/` folder, then:

### Train on CICIDS2017 (recommended — your next step)
```bash
python train_local.py --dataset cicids
```

### Train on NSL-KDD
```bash
python train_local.py --dataset nslkdd
```

### Train on both datasets
```bash
python train_local.py --dataset both
```

### Resume after stopping (auto-detects where you left off)
```bash
python train_local.py --dataset cicids
```
Just run the **exact same command again** — it will skip completed models and
resume any interrupted model from the last completed epoch.

### Skip preprocessing (if you already ran it once)
```bash
python train_local.py --dataset cicids --skip-preprocess
```
Preprocessed data is cached as `.npz` files so the massive CICIDS2017 CSV
loading only happens once.

### Reduce epochs (for quick test)
```bash
python train_local.py --dataset cicids --epochs 10
```

### Reset everything and start fresh
```bash
python train_local.py --dataset cicids --reset
```

---

## 4. How Checkpointing Works

A file `checkpoints/training_state.json` tracks progress automatically.

| File | Purpose |
|------|---------|
| `checkpoints/training_state.json` | JSON tracking which models are done + epoch count |
| `checkpoints/{dataset}_{model}_best.weights.h5` | Best weights saved each epoch |
| `models/bilstm_cic.keras` | Final saved BiLSTM model |
| `models/cnn_cic.keras` | Final saved CNN model |
| `models/transformer_cic.keras` | Final saved Transformer model |
| `models/vae_cic/` | Final VAE (3 files: full, encoder, decoder) |
| `models/ensemble_cic/` | XGBoost meta-classifier |
| `data/processed/cicids/cicids_arrays.npz` | Cached preprocessed arrays |

**Example state file:**
```json
{
  "cicids_bilstm":      { "status": "done",    "epochs_done": 47 },
  "cicids_cnn":         { "status": "pending",  "epochs_done": 12 },
  "cicids_transformer": { "status": "pending",  "epochs_done": 0  },
  "cicids_vae":         { "status": "pending",  "epochs_done": 0  }
}
```
When you restart, `bilstm` is skipped (done), `cnn` resumes from epoch 12,
and the rest start fresh.

---

## 5. Recommended Training Strategy (CICIDS2017 is huge)

CICIDS2017 has ~3.8M samples — each epoch for BiLSTM takes ~20-40 min on CPU.
Here's a suggested schedule:

| Session | What to do |
|---------|-----------|
| Day 1 morning | Start BiLSTM (let it run, early stopping usually triggers ~30-50 epochs) |
| Day 1 evening | Check progress, BiLSTM likely done. CNN starts automatically. |
| Day 2 | CNN finishes, Transformer trains |
| Day 3 | VAE trains (fastest — normal-only subset), then ensemble (XGBoost, ~5 min) |

---

## 6. Outputs

After training completes:
- `outputs/cicids/` — confusion matrices, ROC curves, metrics bar chart, CSV table
- `outputs/nslkdd/` — same for NSL-KDD

---

## 7. Troubleshooting

**Out of memory (RAM):** CICIDS2017 after SMOTE can use ~8–12 GB RAM.
Close other programs. If still failing, reduce batch size in `train_local.py`
(`DEFAULT_BILSTM_PARAMS['batch_size'] = 128`).

**Training too slow:** Set `--epochs 30` and let early stopping decide.
The models converge well before 100 epochs in most cases.

**Checkpoint not resuming:** Check `checkpoints/training_state.json` — if
`epochs_done` is 0, the checkpoint weights file may be missing. Delete the
state entry for that model and restart.
