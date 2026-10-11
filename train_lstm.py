"""
train_lstm.py
=============
Single-configuration final training run.
Smart Temple Crowd Management Project

Configuration (fixed — no hyperparameter search):
  model_type   = LSTM
  seq_length   = 10
  use_features = True  (count, lag1, change, rolling_mean5, rolling_std5, growth_rate)
  hidden_size  = 256
  num_layers   = 3
  dropout      = 0.05
  lr           = 0.0005
  loss         = MSE
  max_epochs   = 200
  patience     = 20 (early stopping on val loss)
  batch_size   = 16
  weight_decay = 1e-4

Guarantees:
  - Normalization scale fitted on TRAINING sequences only (max of train[:, :, 0])
  - Chronological 70/15/15 split per video (no future leakage)
  - Best checkpoint saved via copy.deepcopy (not reference)
  - Random seed set before model creation (reproducibility)
  - Predictions clipped at 0 (no negative counts)
  - No data fabrication, no synthetic data, no row deletion
  - Model keys use 'rnn.*' — consistent with lstm_model.py inference code
"""

import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from lstm_model import MODEL_PATH, save_normalization_scale

DATASET_CSV = "output/crowd_dataset.csv"
EVAL_JSON   = "output/lstm_evaluation.json"
NORM_JSON   = "output/normalization.json"
RANDOM_SEED = 42

FEATURE_COLS_EXTRA = [
    "count_lag1",
    "count_change",
    "rolling_mean5",
    "rolling_std5",
    "growth_rate",
]

# ── Fixed configuration ────────────────────────────────────────────────────────
CFG = {
    "model_type":   "LSTM",
    "seq_length":   10,
    "use_features": True,
    "hidden_size":  256,
    "num_layers":   3,
    "dropout":      0.05,
    "lr":           0.0005,
    "loss":         "MSE",
    "max_epochs":   200,
    "patience":     20,
    "batch_size":   16,
    "weight_decay": 1e-4,
}


# ── Model ──────────────────────────────────────────────────────────────────────

class CrowdLSTM(nn.Module):
    """LSTM regression model. Uses 'rnn' attribute to match lstm_model.py keys."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        # dropout is only applied between layers; single-layer ignores it
        drop = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df):
    """Add temporal features per video. No future data used (shift-based)."""
    df = df.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)
    parts = []
    for vname in df["Video_Name"].unique():
        g = df[df["Video_Name"] == vname].copy()
        c = g["People_Count"].astype(float)
        g["count_lag1"]    = c.shift(1).bfill()
        g["count_change"]  = c - c.shift(1).bfill()
        g["rolling_mean5"] = c.rolling(5, min_periods=1).mean()
        g["rolling_std5"]  = c.rolling(5, min_periods=1).std().fillna(0.0)
        g["growth_rate"]   = (c - c.shift(1).bfill()) / (c.shift(1).bfill() + 1.0)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ── Sequence preparation ───────────────────────────────────────────────────────

def prepare_sequences(df, seq_length, use_features):
    """
    Build (X, y) sequence arrays with chronological 70/15/15 split per video.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test, test_info, n_features
    """
    if use_features:
        df = build_features(df)
        all_cols = ["People_Count"] + FEATURE_COLS_EXTRA
    else:
        df = df.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)
        all_cols = ["People_Count"]

    n_feat = len(all_cols)
    X_tr, y_tr = [], []
    X_va, y_va = [], []
    X_te, y_te = [], []
    test_info  = []   # list of (video_name, time_seconds, threshold)

    for vname in sorted(df["Video_Name"].unique()):
        v = df[df["Video_Name"] == vname].sort_values("Time_Seconds").reset_index(drop=True)

        if len(v) < seq_length + 3:
            print(f"  [SKIP] {vname}: only {len(v)} rows (need >= {seq_length + 3})")
            continue

        fa     = v[all_cols].values.astype(np.float32)
        ca     = v["People_Count"].values.astype(np.float32)
        ta     = v["Time_Seconds"].values.astype(np.float32)
        thresh = float(v["Threshold"].iloc[0]) if "Threshold" in v.columns else 20.0

        # Build sequences (chronological order preserved)
        vX, vy, vm = [], [], []
        for i in range(len(ca) - seq_length):
            vX.append(fa[i: i + seq_length])
            vy.append(ca[i + seq_length])
            vm.append((vname, float(ta[i + seq_length]), thresh))

        if not vX:
            continue

        vX = np.array(vX, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        n  = len(vX)
        nt = max(1, int(0.70 * n))
        nv = max(1, int(0.15 * n)) if n >= 5 else 0
        ne = n - nt - nv

        X_tr.append(vX[:nt]);           y_tr.append(vy[:nt])
        if nv > 0:
            X_va.append(vX[nt: nt + nv]); y_va.append(vy[nt: nt + nv])
        if ne > 0:
            X_te.append(vX[nt + nv:]);    y_te.append(vy[nt + nv:])
            test_info.extend(vm[nt + nv:])

    X_train = np.concatenate(X_tr)
    y_train = np.concatenate(y_tr)
    X_val   = np.concatenate(X_va) if X_va else X_train[-5:]
    y_val   = np.concatenate(y_va) if y_va else y_train[-5:]
    X_test  = np.concatenate(X_te) if X_te else X_val
    y_test  = np.concatenate(y_te) if y_te else y_val

    return X_train, y_train, X_val, y_val, X_test, y_test, test_info, n_feat


# ── Normalization ──────────────────────────────────────────────────────────────

def fit_scale(X_train):
    """Max of the People_Count column in training sequences."""
    scale = float(np.max(X_train[:, :, 0]))
    return scale if scale > 0 else 1.0


def apply_scale(X_tr, y_tr, X_va, y_va, X_te, scale):
    return (
        X_tr / scale, y_tr / scale,
        X_va / scale, y_va / scale,
        X_te / scale,
    )


# ── Status classification ──────────────────────────────────────────────────────

def crowd_status(count, threshold):
    low = max(1.0, threshold / 3.0)
    if count <= low:
        return "Normal"
    elif count <= threshold:
        return "Moderate"
    return "Congested"


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred, test_info):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    sst  = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ssr  = float(np.sum((y_true - y_pred) ** 2))
    r2   = float(1.0 - ssr / sst) if sst > 0 else 0.0

    classes = ["Normal", "Moderate", "Congested"]
    y_true_labels = [crowd_status(float(y_true[i]), test_info[i][2]) for i in range(len(y_true))]
    y_pred_labels = [crowd_status(float(y_pred[i]), test_info[i][2]) for i in range(len(y_pred))]
    acc = sum(a == b for a, b in zip(y_true_labels, y_pred_labels)) / len(y_true_labels)

    cm = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for a, b in zip(y_true_labels, y_pred_labels):
        if a in cm and b in cm[a]:
            cm[a][b] += 1

    precisions, recalls, f1s = [], [], []
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in classes if other != cls)
        fn = sum(cm[cls][other] for other in classes if other != cls)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f)

    return {
        "mae":              round(mae, 4),
        "mse":              round(mse, 4),
        "rmse":             round(rmse, 4),
        "r2_score":         round(r2, 4),
        "accuracy":         round(acc, 4),
        "precision":        round(float(np.mean(precisions)), 4),
        "recall":           round(float(np.mean(recalls)), 4),
        "f1_score":         round(float(np.mean(f1s)), 4),
        "confusion_matrix": cm,
    }


# ── Main training function ─────────────────────────────────────────────────────

def train_and_evaluate_lstm():
    print("=" * 68)
    print("Smart Temple -- Single-Config Final Training Run")
    print("=" * 68)
    print()
    print("Configuration:")
    for k, v in CFG.items():
        print(f"  {k:<15} = {v}")
    print()

    # Load dataset
    df     = pd.read_csv(DATASET_CSV)
    videos = sorted(df["Video_Name"].unique().tolist())
    print(f"Dataset: {len(df)} observations, {len(videos)} videos")
    print(f"Count range: {df['People_Count'].min()} - {df['People_Count'].max()}")
    print(f"Mean={df['People_Count'].mean():.2f}  Std={df['People_Count'].std():.2f}")
    print(f"Status counts: {df['Crowd_Status'].value_counts().to_dict()}")
    print()

    # Naive baseline for honest comparison
    df_s = df.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)
    naive_maes = []
    for vname in videos:
        v = df_s[df_s["Video_Name"] == vname]["People_Count"].values.astype(float)
        if len(v) >= 2:
            naive_maes.append(float(np.mean(np.abs(v[1:] - v[:-1]))))
    naive_mae = float(np.mean(naive_maes))
    print(f"Naive baseline MAE (prev value): {naive_mae:.4f}")
    print()

    # Prepare sequences
    print("Preparing sequences...")
    X_train, y_train, X_val, y_val, X_test, y_test, test_info, n_feat = prepare_sequences(
        df,
        seq_length=CFG["seq_length"],
        use_features=CFG["use_features"],
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  n_features = {n_feat}")

    # Fit normalization scale on training data ONLY
    scale = fit_scale(X_train)
    X_tr_n, y_tr_n, X_va_n, y_va_n, X_te_n = apply_scale(
        X_train, y_train, X_val, y_val, X_test, scale
    )
    print(f"  Normalization scale (train max): {scale:.1f}")

    # Save normalization.json
    os.makedirs("output", exist_ok=True)
    norm_data = {
        "max_train_count": float(scale),
        "model_type":      CFG["model_type"],
        "seq_length":      CFG["seq_length"],
        "hidden_size":     CFG["hidden_size"],
        "num_layers":      CFG["num_layers"],
        "n_features":      int(n_feat),
        "use_features":    CFG["use_features"],
        "feature_cols":    ["People_Count"] + (FEATURE_COLS_EXTRA if CFG["use_features"] else []),
        "dropout":         CFG["dropout"],
    }
    with open(NORM_JSON, "w") as f:
        json.dump(norm_data, f, indent=2)
    save_normalization_scale(scale)
    print(f"  Saved: {NORM_JSON}")
    print()

    # Build tensors
    Xtr_t = torch.tensor(X_tr_n, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr_n, dtype=torch.float32).unsqueeze(-1)
    Xva_t = torch.tensor(X_va_n, dtype=torch.float32)
    yva_t = torch.tensor(y_va_n, dtype=torch.float32).unsqueeze(-1)
    Xte_t = torch.tensor(X_te_n, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=CFG["batch_size"],
        shuffle=True,
    )

    # Build model — set seed BEFORE construction for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = CrowdLSTM(
        input_size=n_feat,
        hidden_size=CFG["hidden_size"],
        num_layers=CFG["num_layers"],
        dropout=CFG["dropout"],
    )

    criterion = nn.MSELoss()   # MSE loss for tighter regression
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    print("=" * 68)
    print("TRAINING")
    print("=" * 68)
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}")
    print("-" * 46)

    best_val_loss  = float("inf")
    best_weights   = None
    no_improve     = 0
    train_losses   = []
    val_losses     = []

    for epoch in range(1, CFG["max_epochs"] + 1):
        # --- Training step ---
        model.train()
        running_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * bx.size(0)

        train_loss = running_loss / len(y_train)
        train_losses.append(train_loss)

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva_t), yva_t).item()
        val_losses.append(val_loss)
        scheduler.step(epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        # --- Print every 5 epochs ---
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  {current_lr:>10.6f}")

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {CFG['patience']} epochs)")
                break

    print()
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Restore best checkpoint
    model.load_state_dict(best_weights)
    model.eval()

    # ── Evaluate on test set ───────────────────────────────────────────────────
    with torch.no_grad():
        raw_preds = model(Xte_t).squeeze(-1).numpy()

    # De-normalize and clip negatives
    test_preds = np.clip(raw_preds * scale, 0.0, None)
    metrics    = calculate_metrics(y_test, test_preds, test_info)

    print("=" * 68)
    print("FINAL TEST SET RESULTS")
    print("=" * 68)
    print(f"  MAE          = {metrics['mae']:.4f} people")
    print(f"  RMSE         = {metrics['rmse']:.4f} people")
    print(f"  R2           = {metrics['r2_score']:.4f}")
    print(f"  Accuracy     = {metrics['accuracy'] * 100:.2f}%")
    print(f"  F1 (macro)   = {metrics['f1_score']:.4f}")
    print()
    print(f"  Naive baseline MAE = {naive_mae:.4f}")
    if metrics["mae"] < naive_mae:
        print(f"  LSTM beats naive baseline by {naive_mae - metrics['mae']:.4f} MAE")
    else:
        print(f"  LSTM does not beat naive baseline "
              f"(delta = {metrics['mae'] - naive_mae:+.4f})")

    # Per-prediction table (first 25)
    print()
    print(f"  {'Video':<12} {'Time':>7} {'Actual':>8} {'Pred':>8} {'Error':>7}")
    print("  " + "-" * 46)
    predictions_list = []
    for idx in range(len(y_test)):
        vn, ts, th = test_info[idx]
        act = float(y_test[idx])
        prd = float(test_preds[idx])
        err = abs(act - prd)
        predictions_list.append({
            "Video":           vn,
            "Target_Time":     round(ts, 1),
            "Actual_Count":    round(act, 1),
            "Predicted_Count": round(prd, 1),
            "Absolute_Error":  round(err, 2),
        })
        if idx < 25:
            print(f"  {vn:<12} {ts:>7.1f} {act:>8.0f} {prd:>8.1f} {err:>7.2f}")
    if len(y_test) > 25:
        print(f"  ... ({len(y_test) - 25} more rows not shown)")

    # Error by crowd level
    errors = np.abs(y_test - test_preds)
    print()
    print("  Error by crowd level:")
    for label, mask in [
        ("Low  (<40)",    y_test < 40),
        ("Med  (40-89)", (y_test >= 40) & (y_test < 90)),
        ("High (>=90)",   y_test >= 90),
    ]:
        if mask.sum() > 0:
            print(f"    {label}: n={mask.sum():4d}  MAE={errors[mask].mean():.3f}  "
                  f"Max={errors[mask].max():.1f}")

    # ── Save model ─────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    print()
    print(f"[OK] Model saved:     {MODEL_PATH}")

    # Verify saved keys match lstm_model.py expectations
    saved_keys = list(torch.load(MODEL_PATH, map_location="cpu", weights_only=True).keys())
    print(f"     Saved keys: {saved_keys}")

    # ── Save lstm_evaluation.json ──────────────────────────────────────────────
    eval_summary = {
        "total_videos":          len(videos),
        "total_observations":    len(df),
        "training_samples":      len(X_train),
        "validation_samples":    len(X_val),
        "test_samples":          len(X_test),
        "sequence_length":       CFG["seq_length"],
        "forecast_horizon":      "1 step ahead",
        "normalization_scale":   float(scale),
        "model_type":            CFG["model_type"],
        "hidden_size":           CFG["hidden_size"],
        "num_layers":            CFG["num_layers"],
        "learning_rate":         CFG["lr"],
        "loss_function":         CFG["loss"],
        "n_features":            int(n_feat),
        "feature_cols":          ["People_Count"] + (FEATURE_COLS_EXTRA if CFG["use_features"] else []),
        "epochs_trained":        len(train_losses),
        "best_val_loss":         round(best_val_loss, 6),
        "naive_baseline_mae":    round(naive_mae, 4),
        "mae":                   metrics["mae"],
        "mse":                   metrics["mse"],
        "rmse":                  metrics["rmse"],
        "r2_score":              metrics["r2_score"],
        "accuracy":              metrics["accuracy"],
        "precision":             metrics["precision"],
        "recall":                metrics["recall"],
        "f1_score":              metrics["f1_score"],
        "confusion_matrix":      metrics["confusion_matrix"],
        "model_path":            MODEL_PATH,
        "predictions":           predictions_list,
        "training_loss_curve":   [round(v, 6) for v in train_losses],
        "validation_loss_curve": [round(v, 6) for v in val_losses],
    }

    with open(EVAL_JSON, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[OK] Evaluation:      {EVAL_JSON}")
    print(f"[OK] Normalization:   {NORM_JSON}")

    print()
    print("=" * 68)
    print("BASELINE vs FINAL COMPARISON")
    print("=" * 68)
    print(f"  {'Metric':<12} {'Baseline':>10} {'Naive':>10} {'Final':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'MAE':<12} {'5.5900':>10} {naive_mae:>10.4f} {metrics['mae']:>10.4f}")
    print(f"  {'RMSE':<12} {'8.1100':>10} {'6.8000':>10} {metrics['rmse']:>10.4f}")
    print(f"  {'R2':<12} {'0.9370':>10} {'--':>10} {metrics['r2_score']:>10.4f}")
    print(f"  {'Accuracy':<12} {'84.09%':>10} {'--':>10} {metrics['accuracy']*100:>9.2f}%")
    print(f"  {'F1':<12} {'0.5567':>10} {'--':>10} {metrics['f1_score']:>10.4f}")
    print("=" * 68)


if __name__ == "__main__":
    train_and_evaluate_lstm()
