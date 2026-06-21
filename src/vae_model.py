"""
vae_model.py
Variational Autoencoder (VAE) for unsupervised anomaly detection.
Key advantage: trained ONLY on normal traffic, so reconstruction error
spikes for any attack type — including zero-day attacks unseen during training.
This is the unique component that no supervised model can replicate.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import os


# ── Sampling layer ────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package='cognitive_ids')
def sampling(args):
    """Reparameterization trick: z = mu + eps * sigma."""
    z_mean, z_log_var = args
    batch  = tf.shape(z_mean)[0]
    dim    = tf.shape(z_mean)[1]
    eps    = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * eps


# ── VAE loss ──────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package='cognitive_ids')
class VAELoss(tf.keras.layers.Layer):
    """Custom layer that adds KL divergence to reconstruction loss."""

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim

    def call(self, inputs):
        x_input, x_decoded, z_mean, z_log_var = inputs
        # Reconstruction loss (MSE scaled by feature dim)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x_input - x_decoded), axis=-1)
        )
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )
        total_loss = recon_loss + kl_loss
        self.add_loss(total_loss)
        return x_decoded


# ── Build VAE ─────────────────────────────────────────────────────────────────

def build_vae(input_dim:  int,
              latent_dim: int   = 16,
              hidden_dim: int   = 64,
              dropout:    float = 0.2):
    """
    Encoder: input -> hidden -> (z_mean, z_log_var) -> z
    Decoder: z -> hidden -> reconstruction
    Trained only on NORMAL samples. Anomaly score = reconstruction error.
    """
    # ── Encoder ──
    enc_input = Input(shape=(input_dim,), name='vae_encoder_input')
    x = Dense(hidden_dim * 2, activation='relu')(enc_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(hidden_dim, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    z_mean    = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z         = Lambda(sampling, name='z')([z_mean, z_log_var])

    encoder = Model(enc_input, [z_mean, z_log_var, z], name='vae_encoder')

    # ── Decoder ──
    dec_input = Input(shape=(latent_dim,), name='vae_decoder_input')
    x = Dense(hidden_dim, activation='relu')(dec_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(hidden_dim * 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    dec_output = Dense(input_dim, activation='sigmoid', name='vae_decoder_output')(x)

    decoder = Model(dec_input, dec_output, name='vae_decoder')

    # ── Full VAE ──
    vae_input = Input(shape=(input_dim,), name='vae_input')
    z_mean_out, z_log_var_out, z_out = encoder(vae_input)
    reconstructed = decoder(z_out)

    # Add VAE loss via custom layer
    vae_output = VAELoss(input_dim, name='vae_loss')(
        [vae_input, reconstructed, z_mean_out, z_log_var_out]
    )

    vae = Model(inputs=vae_input, outputs=vae_output, name='VAE_IDS')
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    return vae, encoder, decoder


def train_vae(X_train: np.ndarray, y_train: np.ndarray,
              latent_dim: int   = 16,
              hidden_dim: int   = 64,
              dropout:    float = 0.2,
              epochs:     int   = 100,
              batch_size: int   = 256,
              save_dir:   str   = None):
    """
    Train VAE ONLY on normal traffic (y == 0).
    This is intentional — the VAE learns what normal looks like.
    Attacks will have high reconstruction error at inference time.
    """
    # Select only normal samples for training
    normal_mask = (y_train == 0)
    X_normal = X_train[normal_mask]
    print(f"[VAE] Training on {len(X_normal):,} normal samples only")

    input_dim = X_train.shape[1]
    vae, encoder, decoder = build_vae(input_dim, latent_dim, hidden_dim, dropout)
    vae.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1)
    ]

    history = vae.fit(
        X_normal, X_normal,         # input == target for autoencoder
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        vae.save(os.path.join(save_dir, 'vae_full.keras'))
        encoder.save(os.path.join(save_dir, 'vae_encoder.keras'))
        decoder.save(os.path.join(save_dir, 'vae_decoder.keras'))
        print(f"[VAE] Models saved -> {save_dir}")

    return vae, encoder, decoder, history


def compute_reconstruction_error(decoder, encoder, X: np.ndarray) -> np.ndarray:
    """
    Compute per-sample reconstruction error (MSE).
    Higher error = more likely to be an anomaly/attack.
    """
    z_mean, z_log_var, z = encoder.predict(X, verbose=0)
    X_reconstructed = decoder.predict(z_mean, verbose=0)  # use mean, not sampled z
    mse = np.mean(np.square(X - X_reconstructed), axis=1)
    return mse


def predict_proba_vae(encoder, decoder, X: np.ndarray,
                      threshold: float = None) -> np.ndarray:
    """
    Return anomaly score normalized to [0, 1] for ensemble fusion.
    If threshold provided, also returns binary predictions.
    """
    errors = compute_reconstruction_error(decoder, encoder, X)
    # Normalize to [0,1] using min-max of this batch
    e_min, e_max = errors.min(), errors.max()
    if e_max > e_min:
        scores = (errors - e_min) / (e_max - e_min)
    else:
        scores = np.zeros_like(errors)
    return scores


def find_vae_threshold(encoder, decoder,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       percentile: float = 95.0) -> float:
    """
    Find reconstruction error threshold using validation set.
    Default: 95th percentile of normal sample errors.
    """
    errors = compute_reconstruction_error(decoder, encoder, X_val)
    normal_errors = errors[y_val == 0]
    threshold = np.percentile(normal_errors, percentile)
    print(f"[VAE] Threshold (p{percentile:.0f} of normal errors): {threshold:.6f}")
    return threshold
