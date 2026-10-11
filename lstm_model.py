"""
lstm_model.py
=============
Shared model definitions and inference utilities.
Smart Temple Crowd Management Project

IMPORTANT: Inference uses normalization.json written during training.
  - max_train_count: scale for de-normalizing predictions
  - model_type: LSTM or GRU
  - seq_length, hidden_size, num_layers: model architecture
  - n_features: input feature count (1 or 6)
  - feature_cols: ordered list of features
"""

import os, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

MODEL_PATH     = "output/lstm_model.pth"
NORM_JSON_PATH = "output/normalization.json"
SEQ_LENGTH     = 10       # default fallback
_FALLBACK_MAX  = 157.0    # matches training max for existing dataset


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class CrowdLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        drop = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, dropout=drop)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        return self.fc(out[:, -1, :])


class CrowdGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        drop = dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=drop)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


# ============================================================
# NORMALIZATION PERSISTENCE
# ============================================================

def load_normalization_config():
    """Load full model config from normalization.json (written during training)."""
    defaults = {
        "max_train_count": _FALLBACK_MAX,
        "model_type":      "LSTM",
        "seq_length":      SEQ_LENGTH,
        "hidden_size":     64,
        "num_layers":      2,
        "n_features":      1,
        "use_features":    False,
        "feature_cols":    ["People_Count"],
    }
    if os.path.exists(NORM_JSON_PATH):
        try:
            with open(NORM_JSON_PATH, "r") as f:
                data = json.load(f)
            # Merge: only override defaults with valid values
            for k in defaults:
                if k in data and data[k] is not None:
                    defaults[k] = data[k]
        except Exception:
            pass
    # Ensure max_train_count is a positive float
    if not isinstance(defaults["max_train_count"], (int, float)) or defaults["max_train_count"] <= 0:
        defaults["max_train_count"] = _FALLBACK_MAX
    return defaults


def load_normalization_scale():
    """Backward-compatible: return just the scale value."""
    return float(load_normalization_config()["max_train_count"])


def save_normalization_scale(max_count):
    """Save training normalization scale (simple form — preserves existing keys if present)."""
    os.makedirs("output", exist_ok=True)
    existing = {}
    if os.path.exists(NORM_JSON_PATH):
        try:
            with open(NORM_JSON_PATH, "r") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing["max_train_count"] = float(max_count)
    with open(NORM_JSON_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def scale_data(data, max_count=None):
    if max_count is None:
        max_count = load_normalization_scale()
    return np.array(data, dtype=np.float32) / max_count


def inverse_scale_data(scaled_data, max_count=None):
    if max_count is None:
        max_count = load_normalization_scale()
    return np.clip(np.round(np.array(scaled_data) * max_count), 0, None).astype(int)


# ============================================================
# MODEL LOADING
# ============================================================

def load_trained_model():
    """Load pre-trained model using architecture from normalization.json."""
    cfg = load_normalization_config()
    mtype = cfg.get("model_type", "LSTM")
    hs    = int(cfg.get("hidden_size", 64))
    nl    = int(cfg.get("num_layers", 2))
    nf    = int(cfg.get("n_features", 1))

    MC    = CrowdLSTM if mtype == "LSTM" else CrowdGRU
    model = MC(input_size=nf, hidden_size=hs, num_layers=nl, output_size=1, dropout=0.2)

    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except Exception:
            try:
                state = torch.load(MODEL_PATH, map_location="cpu")
                model.load_state_dict(state)
            except Exception as e:
                print(f"[WARNING] Could not load model weights: {e}")

    model.eval()
    return model


# ============================================================
# INFERENCE: PREDICT FUTURE CROWD
# ============================================================

def _build_feature_sequence(history_counts, seq_length):
    """
    Build a (seq_length, 6) feature matrix from raw count history.
    Features: [count, lag1, change, rolling_mean5, rolling_std5, growth_rate]
    All features use only past data — no leakage.
    """
    arr = np.array(history_counts, dtype=np.float32)

    # Use last seq_length values (pad with edge if needed)
    if len(arr) < seq_length:
        arr = np.pad(arr, (seq_length - len(arr), 0), mode="edge")
    else:
        arr = arr[-seq_length:]

    n = len(arr)
    features = np.zeros((n, 6), dtype=np.float32)

    for i in range(n):
        count   = arr[i]
        lag1    = arr[i - 1] if i > 0 else arr[0]
        change  = count - lag1
        window  = arr[max(0, i - 4) : i + 1]
        rmean5  = float(np.mean(window))
        rstd5   = float(np.std(window)) if len(window) > 1 else 0.0
        growth  = change / (lag1 + 1.0)

        features[i] = [count, lag1, change, rmean5, rstd5, growth]

    return features   # shape (seq_length, 6)


def predict_future_crowd(history_counts, model=None, forecast_horizon=1):
    """
    Given past crowd counts, predict future crowd count.
    Uses the SAME normalization and model architecture as training.
    No hard-coded scale, no hard-coded ceiling.
    """
    if len(history_counts) == 0:
        return 0

    cfg        = load_normalization_config()
    max_count  = float(cfg["max_train_count"])
    seq_length = int(cfg.get("seq_length", SEQ_LENGTH))
    n_features = int(cfg.get("n_features", 1))
    use_feat   = bool(cfg.get("use_features", False))

    if model is None:
        model = load_trained_model()

    # Build input feature matrix
    if use_feat and n_features > 1:
        feat_matrix = _build_feature_sequence(history_counts, seq_length)  # (seq, 6)
    else:
        # Single-feature fallback
        arr = np.array(history_counts, dtype=np.float32)
        if len(arr) < seq_length:
            arr = np.pad(arr, (seq_length - len(arr), 0), mode="edge")
        else:
            arr = arr[-seq_length:]
        feat_matrix = arr.reshape(-1, 1)

    # Normalize
    feat_norm = feat_matrix / max_count

    # Build tensor: shape (1, seq_length, n_features)
    input_tensor = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        raw_pred = model(input_tensor).item() * max_count

    # Non-negative clip only — no artificial upper limit
    final_count = int(np.clip(np.round(raw_pred), 0, None))
    return final_count
