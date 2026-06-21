"""
bilstm_model.py
Bidirectional LSTM model for network intrusion detection.
Reads sequences in both temporal directions — captures attack patterns
that only make sense with forward + backward context.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, Conv1D, MaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os


def build_bilstm(input_dim: int,
                 lstm_units: int = 128,
                 dense_units: int = 64,
                 dropout_rate: float = 0.3,
                 l2_reg: float = 1e-4) -> Model:
    """
    Architecture: Conv1D (local patterns) -> BiLSTM -> Dense -> Output
    Conv1D before BiLSTM reduces sequence noise and speeds up training.
    L2 regularization + Dropout + BatchNorm prevent overfitting.
    """
    inp = Input(shape=(input_dim, 1), name='bilstm_input')

    # Local feature extraction
    x = Conv1D(filters=64, kernel_size=3, padding='same',
               activation='relu', kernel_regularizer=l2(l2_reg))(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)

    # Bidirectional LSTM — captures both past and future context
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True,
             kernel_regularizer=l2(l2_reg),
             recurrent_regularizer=l2(l2_reg))
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Second BiLSTM for deeper temporal modeling
    x = Bidirectional(
        LSTM(lstm_units // 2,
             kernel_regularizer=l2(l2_reg),
             recurrent_regularizer=l2(l2_reg))
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Dense head
    x = Dense(dense_units, activation='relu',
               kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate / 2)(x)

    # Binary output — sigmoid for probability score
    out = Dense(1, activation='sigmoid', name='bilstm_output')(x)

    model = Model(inputs=inp, outputs=out, name='BiLSTM_IDS')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model


def get_callbacks(patience_es: int = 10, patience_lr: int = 5):
    return [
        EarlyStopping(monitor='val_loss', patience=patience_es,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=patience_lr, min_lr=1e-6, verbose=1)
    ]


def train_bilstm(X_train: np.ndarray, y_train: np.ndarray,
                 X_val:   np.ndarray, y_val:   np.ndarray,
                 lstm_units:   int   = 128,
                 dense_units:  int   = 64,
                 dropout_rate: float = 0.3,
                 l2_reg:       float = 1e-4,
                 epochs:       int   = 100,
                 batch_size:   int   = 256,
                 save_path:    str   = None):
    """
    Reshape input to (samples, features, 1) for Conv1D/LSTM,
    build model, train with early stopping, return model + history.
    """
    input_dim = X_train.shape[1]

    # Reshape: (N, F) -> (N, F, 1)
    X_tr = X_train.reshape(-1, input_dim, 1)
    X_vl = X_val.reshape(-1, input_dim, 1)

    model = build_bilstm(input_dim, lstm_units, dense_units,
                         dropout_rate, l2_reg)
    model.summary()

    history = model.fit(
        X_tr, y_train,
        validation_data=(X_vl, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=1
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[BiLSTM] Saved -> {save_path}")

    return model, history


def predict_proba_bilstm(model, X: np.ndarray) -> np.ndarray:
    """Return probability scores (N,) for ensemble fusion."""
    input_dim = X.shape[1]
    X_r = X.reshape(-1, input_dim, 1)
    return model.predict(X_r, verbose=0).flatten()
