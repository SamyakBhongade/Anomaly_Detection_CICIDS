"""
preprocess.py
Full preprocessing pipeline for NSL-KDD and CICIDS2017.
Handles: loading, cleaning, encoding, normalization, SMOTE, train/test split.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── NSL-KDD config ────────────────────────────────────────────────────────────
NSL_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]
NSL_CATEGORICAL = ['protocol_type', 'service', 'flag']

NSL_ATTACK_MAP = {
    'normal':0,
    'neptune':1,'back':1,'land':1,'pod':1,'smurf':1,'teardrop':1,
    'mailbomb':1,'apache2':1,'processtable':1,'udpstorm':1,
    'ipsweep':1,'nmap':1,'portsweep':1,'satan':1,'mscan':1,'saint':1,
    'ftp_write':1,'guess_passwd':1,'imap':1,'multihop':1,'phf':1,
    'spy':1,'warezclient':1,'warezmaster':1,'sendmail':1,'named':1,
    'snmpgetattack':1,'snmpguess':1,'xlock':1,'xsnoop':1,'httptunnel':1,
    'buffer_overflow':1,'loadmodule':1,'perl':1,'rootkit':1,
    'ps':1,'sqlattack':1,'xterm':1
}

# ── CICIDS2017 — all 8 daily files ───────────────────────────────────────────
CICIDS_FILES = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_nslkdd(train_path, test_path=None):
    print(f"\n[NSL-KDD] Loading train: {train_path}")
    train = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
    print(f"[NSL-KDD] Train shape: {train.shape}")
    test = None
    if test_path and os.path.exists(test_path):
        print(f"[NSL-KDD] Loading test : {test_path}")
        test = pd.read_csv(test_path, header=None, names=NSL_COLUMNS)
        print(f"[NSL-KDD] Test  shape: {test.shape}")
    return train, test


def load_cicids(data_dir, file_list=None):
    """Load and merge CICIDS2017 CSVs. Strips leading spaces from col names."""
    if file_list is None:
        file_list = CICIDS_FILES
    dfs = []
    for fname in file_list:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"[CICIDS] WARNING skipping (not found): {fname}")
            continue
        print(f"[CICIDS] Loading: {fname}")
        tmp = pd.read_csv(fpath, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        dfs.append(tmp)
    if not dfs:
        raise FileNotFoundError(f"No CICIDS files found in {data_dir}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"[CICIDS] Merged shape: {df.shape}")
    print(f"[CICIDS] Label counts:\n{df['Label'].value_counts()}\n")
    return df


# ── Cleaners ──────────────────────────────────────────────────────────────────

def clean_nslkdd(df):
    df = df.drop(columns=['difficulty'], errors='ignore')
    before = len(df)
    df = df.drop_duplicates()
    print(f"[NSL-KDD] Duplicates removed: {before - len(df)}")
    return df


def clean_cicids(df):
    df['Label'] = df['Label'].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"[CICIDS] Rows dropped (inf/NaN): {before - len(df):,}")
    before = len(df)
    df = df.drop_duplicates()
    print(f"[CICIDS] Rows dropped (duplicates): {before - len(df):,}")
    non_label = [c for c in df.columns if c != 'Label']
    constant_cols = [c for c in non_label if df[c].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"[CICIDS] Constant cols dropped: {constant_cols}")
    print(f"[CICIDS] Clean shape: {df.shape}")
    print(f"[CICIDS] Label dist:\n{df['Label'].value_counts()}\n")
    return df


# ── Encoders ──────────────────────────────────────────────────────────────────

def encode_nslkdd(df, encoders=None, fit=True):
    if encoders is None:
        encoders = {}
    for col in NSL_CATEGORICAL:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: int(le.transform([x])[0]) if x in known else 0
            )
    return df, encoders


def make_binary_nslkdd(df):
    df['label'] = df['label'].map(NSL_ATTACK_MAP).fillna(1).astype(int)
    return df


def make_binary_cicids(df):
    df['Label'] = (df['Label'] != 'BENIGN').astype(int)
    return df


# ── Normalization & SMOTE ─────────────────────────────────────────────────────

def normalize(X_train, X_test, save_path=None):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    if save_path:
        joblib.dump(scaler, save_path)
        print(f"[Scaler] Saved -> {save_path}")
    return X_train, X_test, scaler


def apply_smote(X, y, random_state=42):
    # Fix for Windows 11: joblib's loky backend calls `wmic` via subprocess
    # to count physical cores, which is broken/removed in newer Windows builds.
    # Setting LOKY_MAX_CPU_COUNT bypasses that subprocess call entirely.
    import os as _os
    _os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"[SMOTE] Before: {counts}")
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    counts_after = dict(zip(*np.unique(y_res, return_counts=True)))
    print(f"[SMOTE] After : {counts_after}")
    return X_res, y_res


# ── Main pipelines ────────────────────────────────────────────────────────────

def prepare_nslkdd(train_path, test_path, save_dir,
                   use_smote=True, test_size=0.2, random_state=42):
    """
    Full NSL-KDD pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_cols
    """
    os.makedirs(save_dir, exist_ok=True)

    train_df, test_df = load_nslkdd(train_path, test_path)
    train_df = clean_nslkdd(train_df)
    train_df = make_binary_nslkdd(train_df)
    train_df, encoders = encode_nslkdd(train_df, fit=True)
    joblib.dump(encoders, os.path.join(save_dir, 'nslkdd_encoders.pkl'))

    feature_cols = [c for c in train_df.columns if c != 'label']
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['label'].values.astype(int)

    if test_df is not None:
        test_df = clean_nslkdd(test_df)
        test_df = make_binary_nslkdd(test_df)
        test_df, _ = encode_nslkdd(test_df, encoders=encoders, fit=False)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df['label'].values.astype(int)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size,
            random_state=random_state, stratify=y_train
        )

    X_train, X_test, _ = normalize(
        X_train, X_test,
        save_path=os.path.join(save_dir, 'nslkdd_scaler.pkl')
    )

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    print(f"\n[NSL-KDD] READY  Train:{X_train.shape}  Test:{X_test.shape}")
    print(f"[NSL-KDD] Attack ratio train: {y_train.mean():.3f}")
    return X_train, X_test, y_train, y_test, feature_cols


def prepare_cicids(data_dir, save_dir, file_list=None,
                   use_smote=True, test_size=0.2, random_state=42):
    """
    Full CICIDS2017 pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_cols
    """
    os.makedirs(save_dir, exist_ok=True)

    df = load_cicids(data_dir, file_list)
    df = clean_cicids(df)
    df = make_binary_cicids(df)

    feature_cols = [c for c in df.columns if c != 'Label']
    X = df[feature_cols].values.astype(np.float32)
    y = df['Label'].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    X_train, X_test, _ = normalize(
        X_train, X_test,
        save_path=os.path.join(save_dir, 'cicids_scaler.pkl')
    )

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    print(f"\n[CICIDS] READY  Train:{X_train.shape}  Test:{X_test.shape}")
    print(f"[CICIDS] Attack ratio train: {y_train.mean():.3f}")
    return X_train, X_test, y_train, y_test, feature_cols
