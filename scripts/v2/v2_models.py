"""
v2_models.py
======================
Simple, clean, and fully hyperparameter-tunable model definitions
for TQE regression on tabular features.

Models supported:
    - MLP
    - Residual MLP
    - Lightweight Transformer (tabular self-attention)

All models follow:
    build_xxx(input_dim, **hyperparams)

and are fully compatible with:
    - hyperparameter_search()
    - run_kfold_cv()
    - v2_train_single()
"""

import logging
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Set up logging and warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# ============================================================
# Optimizer builder
# ============================================================
def build_optimizer(learning_rate=1e-3):
    """Return Adam optimizer with configurable learning rate."""
    return optimizers.Adam(learning_rate=learning_rate)


# ============================================================
# Standard MLP
# ============================================================
def build_mlp(input_dim, hidden_dim=128, num_layers=3, dropout=0.2,
              learning_rate=1e-3):
    """
    Simple configurable MLP for tabular regression.

    Args:
        input_dim (int)
        hidden_dim (int)
        num_layers (int)
        dropout (float)
        learning_rate (float)

    Returns:
        keras.Model
    """

    logging.info(f"[MLP] hidden_dim={hidden_dim}, layers={num_layers}, "
                f"dropout={dropout}, lr={learning_rate}")

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for _ in range(num_layers):
        model.add(layers.Dense(hidden_dim, activation="relu"))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1))

    model.compile(
        optimizer=build_optimizer(learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model

# ============================================================
# Residual MLP
# ============================================================
def build_residual_mlp(input_dim, hidden_dim=128, num_layers=2, dropout=0.2,
                       learning_rate=1e-3):
    """
    Residual MLP with skip connections.
    Works well for tabular regression with many samples.

    Args:
        input_dim (int)
        hidden_dim (int)
        num_layers (int)
        dropout (float)
        learning_rate (float)
    """

    logging.info(f"[ResMLP] hidden_dim={hidden_dim}, layers={num_layers}, "
                f"dropout={dropout}, lr={learning_rate}")

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)

    for _ in range(num_layers):
        residual = layers.Dense(hidden_dim, activation="relu")(x)
        residual = layers.Dropout(dropout)(residual)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)

    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=build_optimizer(learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model

# ============================================================
# Lightweight Transformer for tabular data
# ============================================================
def build_transformer(input_dim, d_model=64, num_heads=4, dropout=0.1, learning_rate=1e-3):
    """
    Lightweight Transformer treating feature vector as a token.
    Useful when feature interactions are complex.

    Args:
        input_dim (int)
        d_model (int)
        num_heads (int)
        dropout (float)
        learning_rate (float)
    """

    logging.info(f"[Transformer] d_model={d_model}, heads={num_heads}, "
                f"dropout={dropout}, lr={learning_rate}")

    inputs = layers.Input(shape=(input_dim,))

    # Expand features into a single "token"
    x = layers.Dense(d_model)(inputs)         # (batch, d_model)
    x = layers.Reshape((1, d_model))(x)       # (batch, 1, d_model)

    # Self-attention on the single token
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout
    )(x, x)

    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=build_optimizer(learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ============================================================
# Dispatcher
# ============================================================
def get_model_builder(model_name):
    """
    Return model constructor by string name.
    This is the only function used by CV + train_single.

    Args:
        model_name (str): "mlp", "residual_mlp", or "transformer"

    Returns:
        function(input_dim, **hyperparams)
    """

    if model_name == "mlp":
        return build_mlp
    elif model_name == "residual_mlp":
        return build_residual_mlp
    elif model_name == "transformer":
        return build_transformer
    else:
        raise ValueError(f"Unknown model {model_name}")
