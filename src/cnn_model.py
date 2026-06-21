"""
cnn_model.py
1D CNN model for network intrusion detection.
Captures local spatial patterns (e.g., port scan burst signatures)
that sequence models like LSTM can miss.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization, Add, Activation
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os


def residual_block(x, filters: int, kernel_size: int = 3,
                   l2_reg: float = 1e-4, dropout: float = 0.2):
    """Residual block to help with gradient flow in deeper CNNs."""
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same',
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters, kernel_size, padding='same',
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_cnn(input_dim: int,
              filters:      int   = 64,
              dense_units:  int   = 64,
              dropout_rate: float = 0.3,
              l2_reg:       float = 1e-4) -> Model:
    """
    Architecture: Conv1D stem -> 2 Residual blocks -> GlobalAvgPool -> Dense -> Output
    Residual connections prevent overfitting on deep stacks.
    GlobalAveragePooling avoids the large flatten layer that causes overfitting.
    """
    inp = Input(shape=(input_dim, 1), name='cnn_input')

    # Stem convolution
    x = Conv1D(filters, kernel_size=5, padding='same',
               activation='relu', kernel_regularizer=l2(l2_reg))(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Residual blocks — multi-scale feature extraction
    x = residual_block(x, filters,       l2_reg=l2_reg, dropout=dropout_rate)
    x = residual_block(x, filters * 2,   l2_reg=l2_reg, dropout=dropout_rate)
    x = MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, filters * 2,   l2_reg=l2_reg, dropout=dropout_rate)

    # Global average pooling — much less overfitting than Flatten
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    # Dense head
    x = Dense(dense_units, activation='relu',
               kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate / 2)(x)

    out = Dense(1, activation='sigmoid', name='cnn_output')(x)

    model = Model(inputs=inp, outputs=out, name='CNN_IDS')
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


def train_cnn(X_train: np.ndarray, y_train: np.ndarray,
              X_val:   np.ndarray, y_val:   np.ndarray,
              filters:      int   = 64,
              dense_units:  int   = 64,
              dropout_rate: float = 0.3,
              l2_reg:       float = 1e-4,
              epochs:       int   = 100,
              batch_size:   int   = 256,
              save_path:    str   = None):
    """Train CNN, return model and history."""
    input_dim = X_train.shape[1]

    # Reshape: (N, F) -> (N, F, 1)
    X_tr = X_train.reshape(-1, input_dim, 1)
    X_vl = X_val.reshape(-1, input_dim, 1)

    model = build_cnn(input_dim, filters, dense_units, dropout_rate, l2_reg)
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
        print(f"[CNN] Saved -> {save_path}")

    return model, history


def predict_proba_cnn(model, X: np.ndarray) -> np.ndarray:
    """Return probability scores (N,) for ensemble fusion."""
    input_dim = X.shape[1]
    X_r = X.reshape(-1, input_dim, 1)
    return model.predict(X_r, verbose=0).flatten()
