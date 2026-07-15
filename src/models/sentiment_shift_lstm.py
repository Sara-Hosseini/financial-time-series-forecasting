# src/models/sentiment_shift_lstm.py
# SentimentShift Quantile LSTM
#
# Architecture:
#   Price branch  : LSTM processes the full (window, n_price) sequence -> base Q10/Q50/Q90
#   Sentiment branch: MLP processes last-step (n_sent,) features -> scalar shift
#   Output        : base_quantiles + sentiment_shift  (broadcast over 3 quantiles)
#
# This lets sentiment directly push the predicted price distribution up or down,
# which is the thesis's core distribution-shift hypothesis.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Lambda
from tensorflow.keras.models import Model

# Thesis-aligned feature column definitions
PRICE_FEATURE_COLS = [
    "log_ret_1", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10",
    "hl_range", "vol_10", "vol_20", "rsi_14",
]

CORE_SENTIMENT_COLS = [
    "avg_sentiment_score", "positive_ratio", "negative_ratio", "news_count",
]

# 11 event/fundamental feature columns (9 binary flags + 2 aggregates)
EVENT_FEATURE_COLS = [
    "earnings_event", "analyst_upgrade", "analyst_downgrade",
    "product_launch", "lawsuit_event", "regulatory_event",
    "partnership_event", "management_change_event", "macro_event",
    "event_count", "event_intensity",
]

N_PRICE      = len(PRICE_FEATURE_COLS)    # 9
N_SENT       = len(CORE_SENTIMENT_COLS)   # 4
N_SENT_EVENT = N_SENT + len(EVENT_FEATURE_COLS)  # 4 + 11 = 15  (event-model branch width)


def pinball_loss(quantiles):
    q_tf = tf.constant(list(quantiles), dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.repeat(y_true, len(quantiles), axis=-1)
        errors = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q_tf * errors, (q_tf - 1.0) * errors))
    return loss


def build_sentiment_shift_model(window: int, n_price: int, n_sent: int, cfg: dict) -> tf.keras.Model:
    """
    Build the SentimentShiftQuantileLSTM.

    Input: (batch, window, n_price + n_sent)  — single combined tensor
    Inside the model the tensor is sliced:
        price_seq  = input[:, :,      :n_price]   (all time steps, price features)
        sent_last  = input[:, -1, n_price:]       (last time step, sentiment features)

    Returns a compiled Keras Model.
    """
    inp = Input(shape=(window, n_price + n_sent), name="combined_input")

    # ── Price branch: LSTM over full sequence ──────────────────────
    price_seq = Lambda(lambda x: x[:, :, :n_price], name="price_slice")(inp)
    x = LSTM(cfg["units1"], return_sequences=True, name="lstm1")(price_seq)
    x = Dropout(cfg["dropout"])(x)
    x = LSTM(cfg["units2"], name="lstm2")(x)
    x = Dropout(cfg["dropout"])(x)
    x = Dense(32, activation="relu", name="dense_price")(x)
    base_quantiles = Dense(3, name="base_quantiles")(x)          # (batch, 3)

    # ── Sentiment branch: MLP on last time-step's sentiment ────────
    sent_last = Lambda(lambda x: x[:, -1, n_price:], name="sent_slice")(inp)
    s = Dense(16, activation="relu", name="sent_dense1")(sent_last)
    s = Dense(8,  activation="relu", name="sent_dense2")(s)
    sentiment_shift = Dense(1, name="sentiment_shift")(s)        # (batch, 1)

    # ── Combine: broadcast shift across Q10, Q50, Q90 ─────────────
    # (batch, 3) + (batch, 1)  ->  (batch, 3)
    final = Lambda(
        lambda inputs: inputs[0] + inputs[1],
        name="combine"
    )([base_quantiles, sentiment_shift])

    model = Model(inputs=inp, outputs=final, name="SentimentShiftQuantileLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"], clipnorm=1.0),
        loss=pinball_loss(cfg["quantiles"]),
    )
    return model


def build_sentiment_shift_event_model(
    window: int, n_price: int, n_sent_event: int, cfg: dict
) -> tf.keras.Model:
    """
    SentimentShiftQuantileLSTM — Event-Enhanced Variant.

    Same dual-branch architecture as build_sentiment_shift_model, but the
    sentiment/event MLP branch is wider: it receives 4 sentiment + 11 event
    features = 15 inputs instead of 4.

    Input: (batch, window, n_price + n_sent_event)
    Sliced internally:
        price_seq   = input[:, :,       :n_price]       (window × 9 price features)
        sent_ev_last = input[:, -1, n_price:]           (15 sentiment+event features,
                                                          last time step only)
    Output: base_quantiles + sentiment_event_shift  →  final Q10/Q50/Q90
    """
    inp = Input(shape=(window, n_price + n_sent_event), name="combined_input_event")

    # ── Price branch (identical to base model) ─────────────────────
    price_seq = Lambda(lambda x: x[:, :, :n_price], name="price_slice")(inp)
    x = LSTM(cfg["units1"], return_sequences=True, name="lstm1")(price_seq)
    x = Dropout(cfg["dropout"])(x)
    x = LSTM(cfg["units2"], name="lstm2")(x)
    x = Dropout(cfg["dropout"])(x)
    x = Dense(32, activation="relu", name="dense_price")(x)
    base_quantiles = Dense(3, name="base_quantiles")(x)              # (batch, 3)

    # ── Sentiment + Event branch: wider MLP ────────────────────────
    sent_ev = Lambda(lambda x: x[:, -1, n_price:], name="sent_event_slice")(inp)
    s = Dense(32, activation="relu", name="sent_event_dense1")(sent_ev)  # wider: 32
    s = Dense(16, activation="relu", name="sent_event_dense2")(s)
    s = Dense(8,  activation="relu", name="sent_event_dense3")(s)
    sentiment_event_shift = Dense(1, name="sentiment_event_shift")(s)   # (batch, 1)

    # ── Combine ────────────────────────────────────────────────────
    final = Lambda(
        lambda inputs: inputs[0] + inputs[1],
        name="combine_event"
    )([base_quantiles, sentiment_event_shift])

    model = Model(
        inputs=inp, outputs=final,
        name="SentimentShiftEventQuantileLSTM"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"], clipnorm=1.0),
        loss=pinball_loss(cfg["quantiles"]),
    )
    return model


def get_model_summary(window: int = 30) -> str:
    cfg = {
        "units1": 64, "units2": 32, "dropout": 0.25,
        "lr": 5e-4, "quantiles": (0.1, 0.5, 0.9),
    }
    model = build_sentiment_shift_model(window, N_PRICE, N_SENT, cfg)
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_model_summary())
