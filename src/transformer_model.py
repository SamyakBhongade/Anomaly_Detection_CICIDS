"""
transformer_model.py
Transformer encoder model for network intrusion detection.
Self-attention captures long-range dependencies across all flow features
simultaneously — something LSTM forgets and CNN cannot reach.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os


class TransformerBlock(tf.keras.layers.Layer):
    """Single Transformer encoder block with multi-head attention + FFN."""

    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Multi-head self-attention with residual
        attn_out = self.att(inputs, inputs)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(inputs + attn_out)

        # Feed-forward network with residual
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layernorm2(out1 + ffn_out)


def build_transformer(input_dim:    int,
                      embed_dim:    int   = 64,
                      num_heads:    int   = 4,
                      ff_dim:       int   = 128,
                      num_blocks:   int   = 2,
                      dense_units:  int   = 64,
                      dropout_rate: float = 0.2) -> Model:
    """
    Architecture:
      Linear projection -> N x TransformerBlock -> GlobalAvgPool -> Dense -> Output
    embed_dim must be divisible by num_heads.
    GlobalAveragePooling aggregates all position outputs without overfitting.
    """
    assert embed_dim % num_heads == 0, \
        f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

    inp = Input(shape=(input_dim, 1), name='transformer_input')

    # Project scalar features to embedding dimension
    x = Dense(embed_dim)(inp)

    # Stack transformer encoder blocks
    for i in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim,
                             dropout_rate, name=f'transformer_block_{i}')(x)

    # Aggregate sequence dimension
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    # Dense classifier head
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate / 2)(x)

    out = Dense(1, activation='sigmoid', name='transformer_output')(x)

    model = Model(inputs=inp, outputs=out, name='Transformer_IDS')
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


def train_transformer(X_train: np.ndarray, y_train: np.ndarray,
                      X_val:   np.ndarray, y_val:   np.ndarray,
                      embed_dim:    int   = 64,
                      num_heads:    int   = 4,
                      ff_dim:       int   = 128,
                      num_blocks:   int   = 2,
                      dense_units:  int   = 64,
                      dropout_rate: float = 0.2,
                      epochs:       int   = 100,
                      batch_size:   int   = 256,
                      save_path:    str   = None):
    """Train Transformer, return model and history."""
    input_dim = X_train.shape[1]

    # Reshape: (N, F) -> (N, F, 1)
    X_tr = X_train.reshape(-1, input_dim, 1)
    X_vl = X_val.reshape(-1, input_dim, 1)

    model = build_transformer(input_dim, embed_dim, num_heads,
                              ff_dim, num_blocks, dense_units, dropout_rate)
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
        print(f"[Transformer] Saved -> {save_path}")

    return model, history


def predict_proba_transformer(model, X: np.ndarray) -> np.ndarray:
    """Return probability scores (N,) for ensemble fusion."""
    input_dim = X.shape[1]
    X_r = X.reshape(-1, input_dim, 1)
    return model.predict(X_r, verbose=0).flatten()
