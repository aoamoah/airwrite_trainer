"""Model constructors for the three compared architectures."""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(params: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth"),
        class_weight=params.get("class_weight", "balanced"),
        n_jobs=params.get("n_jobs", -1),
        random_state=seed,
    )


def build_recurrent(kind: str, params: dict, window: int, n_features: int):
    """Build an LSTM or GRU binary classifier."""
    from tensorflow import keras
    from tensorflow.keras import layers

    rnn_layer = {"lstm": layers.LSTM, "gru": layers.GRU}[kind]
    units = params.get("units", 64)
    dropout = params.get("dropout", 0.3)

    # No Masking layer: an all-zero frame means "no hand visible", which is
    # real not-writing signal — and unmasked sequences let Keras use the
    # fused cuDNN kernels on GPU.
    model = keras.Sequential(
        [
            keras.Input(shape=(window, n_features)),
            rnn_layer(units, return_sequences=True),
            layers.Dropout(dropout),
            rnn_layer(units // 2),
            layers.Dropout(dropout),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name=f"{kind}_w{window}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(params.get("learning_rate", 1e-3)),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
