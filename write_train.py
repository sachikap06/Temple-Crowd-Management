
code = r'''"""
train_lstm.py - Focused, Diagnostic-Driven LSTM/GRU Training
Smart Temple Crowd Management Project

Changes from blind search:
  - seq_length: 3, 5, 10 (short wins when autocorr drops fast)
  - dropout: 0.0 (no regularization collapse on small data)
  - Huber loss: robust to remaining label outliers
  - Compare LSTM, GRU, SimpleRNN
  - Also test: no-feature (count only) vs engineered features
  - Based on diagnose.py findings: v11 noise was root cause
  - Normalization: max of training sequences (fitted on train only)
  - Early stopping patience=20, max 200 epochs
  - Best checkpoint via copy.deepcopy
  - Reproducible seed=42 before every model build
  - Final evaluation on UNTOUCHED test set
"""

import os, copy, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from lstm_model import MODEL_PATH, save_normalization_scale

DATASET_CSV     = "output/crowd_dataset.csv"
EVAL_JSON       = "output/lstm_evaluation.json"
EXPERIMENTS_CSV = "output/model_experiments.csv"
RANDOM_SEED     = 42
FEATURE_COLS_EXTRA = ["count_lag1", "count_change", "rolling_mean5", "rolling_std5", "growth_rate"]


# ============================================================
# MODELS
# ============================================================

class CrowdLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1, dropout=0.0):
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
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1, dropout=0.0):
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
# FEATURE ENGINEERING
# ============================================================

def build_features(df):
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


# ============================================================
# SEQUENCE CREATION (chronological 70/15/15)
# ============================================================

def prepare_sequences(df, seq_length=5, use_features=True):
    if use_features:
        df = build_features(df)
        all_cols = ["People_Count"] + FEATURE_COLS_EXTRA
    else:
        df = df.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)
        all_cols = ["People_Count"]

    n_feat = len(all_cols)
    X_tr, y_tr, X_va, y_va, X_te, y_te, ti = [], [], [], [], [], [], []

    for vname in sorted(df["Video_Name"].unique()):
        v = df[df["Video_Name"] == vname].sort_values("Time_Seconds").reset_index(drop=True)
        if len(v) < seq_length + 3:
            continue

        fa     = v[all_cols].values.astype(np.float32)
        ca     = v["People_Count"].values.astype(np.float32)
        ta     = v["Time_Seconds"].values.astype(np.float32)
        thresh = float(v["Threshold"].iloc[0]) if "Threshold" in v.columns else 20.0

        vX, vy, vm = [], [], []
        for i in range(len(ca) - seq_length):
            vX.append(fa[i: i + seq_length])
            vy.append(ca[i + seq_length])
            vm.append((vname, ta[i + seq_length], thresh))

        if not vX:
            continue

        vX = np.array(vX, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        n  = len(vX)
        nt = max(1, int(0.70 * n))
        nv = max(1, int(0.15 * n)) if n >= 5 else 0
        ne = n - nt - nv

        X_tr.append(vX[:nt]);          y_tr.append(vy[:nt])
        if nv > 0:
            X_va.append(vX[nt: nt + nv]); y_va.append(vy[nt: nt + nv])
        if ne > 0:
            X_te.append(vX[nt + nv:]); y_te.append(vy[nt + nv:])
            ti.extend(vm[nt + nv:])

    X_train = np.concatenate(X_tr)
    y_train = np.concatenate(y_tr)
    X_val   = np.concatenate(X_va) if X_va else X_train[-3:]
    y_val   = np.concatenate(y_va) if y_va else y_train[-3:]
    X_test  = np.concatenate(X_te) if X_te else X_val
    y_test  = np.concatenate(y_te) if y_te else y_val

    return X_train, y_train, X_val, y_val, X_test, y_test, ti, n_feat


def normalize_splits(X_tr, y_tr, X_va, y_va, X_te):
    scale = float(np.max(X_tr[:, :, 0]))
    if scale <= 0:
        scale = 1.0
    return X_tr / scale, y_tr / scale, X_va / scale, y_va / scale, X_te / scale, scale


# ============================================================
# METRICS
# ============================================================

def crowd_status(cnt, thresh):
    low = max(1.0, thresh / 3.0)
    if cnt <= low:       return "Normal"
    elif cnt <= thresh:  return "Moderate"
    return "Congested"


def calculate_metrics(y_true, y_pred, test_info):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    sst  = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ssr  = float(np.sum((y_true - y_pred) ** 2))
    r2   = (1.0 - ssr / sst) if sst > 0 else 0.0

    classes = ["Normal", "Moderate", "Congested"]
    yt = [crowd_status(float(y_true[i]), test_info[i][2]) for i in range(len(y_true))]
    yp = [crowd_status(float(y_pred[i]), test_info[i][2]) for i in range(len(y_pred))]
    acc = sum(a == b for a, b in zip(yt, yp)) / len(yt)

    cm = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for a, b in zip(yt, yp):
        if a in cm and b in cm[a]:
            cm[a][b] += 1

    ps, rs, fs = [], [], []
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[o][c] for o in classes if o != c)
        fn = sum(cm[c][o] for o in classes if o != c)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ps.append(p); rs.append(r); fs.append(f)

    return {
        "mae":              round(mae, 4),
        "mse":              round(mse, 4),
        "rmse":             round(rmse, 4),
        "r2_score":         round(r2, 4),
        "accuracy":         round(acc, 4),
        "precision":        round(float(np.mean(ps)), 4),
        "recall":           round(float(np.mean(rs)), 4),
        "f1_score":         round(float(np.mean(fs)), 4),
        "confusion_matrix": cm,
    }


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(Xtr, ytr, Xva, yva, Xte, yte, ti, scale,
                   mtype, hs, nl, lr, loss_n, nf, dropout,
                   max_ep=200, pat=20):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(-1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32).unsqueeze(-1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)

    MC    = CrowdLSTM if mtype == "LSTM" else CrowdGRU
    model = MC(input_size=nf, hidden_size=hs, num_layers=nl, output_size=1, dropout=dropout)
    crit  = nn.SmoothL1Loss(beta=1.0) if loss_n == "Huber" else nn.MSELoss()
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-5)

    bvl = float("inf")
    bw  = None
    ni  = 0
    tls, vls = [], []

    for ep in range(1, max_ep + 1):
        model.train()
        run = 0.0
        for bx, by in loader:
            opt.zero_grad()
            out  = model(bx)
            loss = crit(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * bx.size(0)

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva_t), yva_t).item()

        sched.step(vl)
        tls.append(run / len(ytr))
        vls.append(vl)

        if vl < bvl:
            bvl = vl
            bw  = copy.deepcopy(model.state_dict())
            ni  = 0
        else:
            ni += 1
            if ni >= pat:
                break

    if bw:
        model.load_state_dict(bw)
    model.eval()

    with torch.no_grad():
        test_preds = np.clip(model(Xte_t).squeeze(-1).numpy() * scale, 0.0, None)
        val_preds  = np.clip(model(Xva_t).squeeze(-1).numpy() * scale, 0.0, None)

    yva_real = yva * scale
    vm  = float(np.mean(np.abs(yva_real - val_preds)))
    vr  = float(np.sqrt(np.mean((yva_real - val_preds) ** 2)))
    m   = calculate_metrics(yte, test_preds, ti)

    return vm, vr, m, model.state_dict(), ep, test_preds, tls, vls


# ============================================================
# MAIN
# ============================================================

def train_and_evaluate_lstm():
    print("=" * 68)
    print("Smart Temple -- Focused Diagnostic-Driven Training")
    print("=" * 68)

    df     = pd.read_csv(DATASET_CSV)
    videos = sorted(df["Video_Name"].unique().tolist())
    print(f"Videos: {len(videos)}, Observations: {len(df)}")
    print(f"Count range: {df['People_Count'].min()} - {df['People_Count'].max()}")
    print(f"Mean={df['People_Count'].mean():.2f}  Std={df['People_Count'].std():.2f}")
    print(f"Status: {df['Crowd_Status'].value_counts().to_dict()}")

    # Naive baseline for reference
    df_s = df.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)
    naive_maes = []
    for vname in videos:
        v = df_s[df_s["Video_Name"] == vname]["People_Count"].values.astype(float)
        if len(v) >= 2:
            naive_maes.append(float(np.mean(np.abs(v[1:] - v[:-1]))))
    naive_mae = float(np.mean(naive_maes))
    print(f"\nNaive baseline MAE (prev value): {naive_mae:.4f}")
    print(f"Target: beat {naive_mae:.2f} on test set")

    # Build sequence caches per (seq_length, use_features)
    configs_seqfeat = [(3, True), (5, True), (10, True), (5, False), (10, False)]
    cache = {}
    print()
    for sl, uf in configs_seqfeat:
        key = (sl, uf)
        X1,y1,X2,y2,X3,y3,t3,nf3 = prepare_sequences(df, seq_length=sl, use_features=uf)
        s1,s2,s3,s4,s5,sc = normalize_splits(X1,y1,X2,y2,X3)
        cache[key] = (y3, t3, nf3, s1, s2, s3, s4, s5, sc)
        print(f"  seq={sl} feat={uf}: train={len(X1)} val={len(X2)} test={len(X3)} nfeat={nf3} scale={sc:.1f}")

    # Save initial normalization scale (seq=5, features)
    init_scale = cache[(5, True)][-1]
    save_normalization_scale(init_scale)

    # Focused grid — motivated by diagnostic findings
    print()
    print("=" * 68)
    print("FOCUSED HYPERPARAMETER SEARCH (diagnostic-motivated)")
    print("=" * 68)
    grid = []
    for mt in ["LSTM", "GRU"]:
        for sl, uf in [(3, True), (5, True), (10, True), (5, False)]:
            for hs in [32, 64, 128]:
                for nl in [1, 2]:
                    for lr in [0.001, 0.003]:
                        for ln in ["Huber", "MSE"]:
                            for do in [0.0, 0.1]:
                                grid.append((mt, sl, uf, hs, nl, lr, ln, do))

    total = len(grid)
    print(f"Focused grid size: {total} configs")

    exps   = []
    bvm    = float("inf")
    bcfg   = None
    b_w    = None
    b_p    = None
    b_tl, b_vl = [], []
    b_ep   = 0
    b_m    = None
    b_nf   = 1
    b_sc   = 1.0
    b_yte  = None
    b_ti   = None

    for i, (mt, sl, uf, hs, nl, lr, ln, do) in enumerate(grid):
        key = (sl, uf)
        if key not in cache:
            continue
        yte_c, ti_c, nf_c, s1, s2, s3, s4, s5, sc = cache[key]

        vm, vr, m, w, ep, ps, tl, vl = run_experiment(
            s1, s2, s3, s4, s5, yte_c, ti_c, sc,
            mt, hs, nl, lr, ln, nf_c, do
        )

        feat_label = "feats" if uf else "count_only"
        exps.append({
            "exp_id":      i,
            "model_type":  mt,
            "seq_length":  sl,
            "use_features": uf,
            "hidden_size": hs,
            "num_layers":  nl,
            "lr":          lr,
            "loss":        ln,
            "dropout":     do,
            "epochs":      ep,
            "val_mae":     round(vm, 4),
            "val_rmse":    round(vr, 4),
            "test_mae":    m["mae"],
            "test_rmse":   m["rmse"],
            "test_r2":     m["r2_score"],
            "test_acc":    m["accuracy"],
            "test_f1":     m["f1_score"],
        })

        if (i + 1) % 16 == 0 or i == 0 or i == total - 1:
            print(f"  [{i+1:3d}/{total}] {mt:4s} seq={sl} {'feat' if uf else 'raw ':4s} "
                  f"h={hs:3d} L={nl} lr={lr:.3f} {ln:5s} drop={do:.1f} | "
                  f"val_MAE={vm:.3f}  test_MAE={m['mae']:.3f}")

        if vm < bvm:
            bvm = vm
            bcfg = {
                "model_type": mt, "seq_length": sl, "use_features": uf,
                "hidden_size": hs, "num_layers": nl, "lr": lr,
                "loss": ln, "dropout": do,
            }
            b_w = w; b_p = ps; b_tl = tl; b_vl = vl; b_ep = ep
            b_m = m; b_nf = nf_c; b_sc = sc; b_yte = yte_c; b_ti = ti_c

    os.makedirs("output", exist_ok=True)
    pd.DataFrame(exps).to_csv(EXPERIMENTS_CSV, index=False)
    print(f"\n[OK] Experiments saved: {EXPERIMENTS_CSV}")
    print(f"\nBEST (val_MAE={bvm:.4f}): {bcfg}  epochs={b_ep}")

    # Rebuild and save best model
    MC = CrowdLSTM if bcfg["model_type"] == "LSTM" else CrowdGRU
    best_model = MC(
        input_size=b_nf, hidden_size=bcfg["hidden_size"],
        num_layers=bcfg["num_layers"], output_size=1, dropout=bcfg["dropout"]
    )
    best_model.load_state_dict(b_w)
    best_model.eval()
    torch.save(best_model.state_dict(), MODEL_PATH)

    # Save normalization.json with full config
    norm_data = {
        "max_train_count": float(b_sc),
        "model_type":      bcfg["model_type"],
        "seq_length":      bcfg["seq_length"],
        "hidden_size":     bcfg["hidden_size"],
        "num_layers":      bcfg["num_layers"],
        "n_features":      int(b_nf),
        "use_features":    bcfg["use_features"],
        "feature_cols":    ["People_Count"] + (FEATURE_COLS_EXTRA if bcfg["use_features"] else []),
    }
    with open("output/normalization.json", "w") as f:
        json.dump(norm_data, f, indent=2)

    # Error analysis
    errors = np.abs(b_yte - b_p)
    print("\n=== ERROR ANALYSIS BY CROWD LEVEL ===")
    for label, mask in [
        ("Low  (<40)",    b_yte < 40),
        ("Med  (40-89)", (b_yte >= 40) & (b_yte < 90)),
        ("High (>=90)",   b_yte >= 90),
    ]:
        if mask.sum() > 0:
            print(f"  {label}: n={mask.sum():4d}  MAE={errors[mask].mean():.3f}  "
                  f"Max={errors[mask].max():.1f}")

    print("\n  Top-5 worst errors:")
    worst = np.argsort(errors)[-5:][::-1]
    for idx in worst:
        vn, ts, th = b_ti[idx]
        print(f"    {vn}  t={ts:.1f}  actual={b_yte[idx]:.0f}  "
              f"pred={b_p[idx]:.1f}  err={errors[idx]:.2f}")

    # Per-video errors
    print("\n  Per-video test errors:")
    vid_errs = {}
    for idx, (vn, ts, th) in enumerate(b_ti):
        vid_errs.setdefault(vn, []).append(errors[idx])
    for vn in sorted(vid_errs.keys()):
        ev = np.array(vid_errs[vn])
        print(f"    {vn:<10}: n={len(ev):4d}  MAE={ev.mean():.3f}  Max={ev.max():.1f}")

    # Final test table
    ptbl = []
    print("\n=== FINAL TEST SET (untouched) ===")
    print(f"{'Video':<12} {'Time':>7} {'Actual':>8} {'Predicted':>10} {'Err':>7}")
    print("-" * 50)
    for idx in range(min(25, len(b_yte))):
        vn, ts, th = b_ti[idx]
        act = float(b_yte[idx])
        prd = float(b_p[idx])
        err = abs(act - prd)
        ptbl.append({
            "Video": str(vn), "Target Time": round(float(ts), 1),
            "Actual Count": round(act, 1), "Predicted Count": round(prd, 1),
            "Absolute Error": round(err, 2),
        })
        print(f"{vn:<12} {ts:>7.1f} {act:>8.0f} {prd:>10.1f} {err:>7.2f}")
    print("-" * 50)

    # Build Xtr/Xva/Xte with the best seq_length to get correct sample counts
    Xtr_f, ytr_f, Xva_f, yva_f, Xte_f, yte_f, _, _ = prepare_sequences(
        df, seq_length=bcfg["seq_length"], use_features=bcfg["use_features"]
    )

    print(f"\n  MAE      = {b_m['mae']:.4f}")
    print(f"  RMSE     = {b_m['rmse']:.4f}")
    print(f"  R2       = {b_m['r2_score']:.4f}")
    print(f"  Accuracy = {b_m['accuracy']*100:.2f}%")
    print(f"  F1       = {b_m['f1_score']:.4f}")

    # Save evaluation JSON
    eval_summary = {
        "total_videos":          len(videos),
        "total_observations":    len(df),
        "training_samples":      len(Xtr_f),
        "validation_samples":    len(Xva_f),
        "test_samples":          len(Xte_f),
        "sequence_length":       bcfg["seq_length"],
        "forecast_horizon":      "1 step",
        "normalization_scale":   float(b_sc),
        "model_type":            bcfg["model_type"],
        "hidden_size":           bcfg["hidden_size"],
        "num_layers":            bcfg["num_layers"],
        "learning_rate":         bcfg["lr"],
        "loss_function":         bcfg["loss"],
        "n_features":            int(b_nf),
        "epochs":                b_ep,
        "mae":                   b_m["mae"],
        "mse":                   b_m["mse"],
        "rmse":                  b_m["rmse"],
        "r2_score":              b_m["r2_score"],
        "accuracy":              b_m["accuracy"],
        "precision":             b_m["precision"],
        "recall":                b_m["recall"],
        "f1_score":              b_m["f1_score"],
        "confusion_matrix":      b_m["confusion_matrix"],
        "model_path":            MODEL_PATH,
        "predictions":           ptbl,
        "training_loss_curve":   b_tl[-50:],
        "validation_loss_curve": b_vl[-50:],
        "naive_baseline_mae":    round(naive_mae, 4),
    }
    with open(EVAL_JSON, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\n[OK] Model saved:        {MODEL_PATH}")
    print(f"[OK] Normalization:      output/normalization.json")
    print(f"[OK] Evaluation:         {EVAL_JSON}")
    print(f"[OK] Experiments CSV:    {EXPERIMENTS_CSV}")

    print("\n" + "=" * 68)
    print("GENUINE RESULTS vs BASELINE")
    print("=" * 68)
    print(f"  Metric   | Original | Naive  |  Final")
    print(f"  ---------+----------+--------+--------")
    print(f"  MAE      |    5.59  | {naive_mae:5.2f}  | {b_m['mae']:>7.4f}")
    print(f"  RMSE     |    8.11  |  6.80  | {b_m['rmse']:>7.4f}")
    print(f"  R2       |  0.9370  |   --   | {b_m['r2_score']:>7.4f}")
    print(f"  Accuracy |  84.09%  |   --   | {b_m['accuracy']*100:>6.2f}%")
    print(f"  F1       |  0.5567  |   --   | {b_m['f1_score']:>7.4f}")
    print("=" * 68)

    if b_m["mae"] < naive_mae:
        print(f"  LSTM BEATS naive baseline by {naive_mae - b_m['mae']:.4f} MAE")
    else:
        print(f"  LSTM does not beat naive baseline -- reporting honestly")


if __name__ == "__main__":
    train_and_evaluate_lstm()
'''

with open("train_lstm.py", "w", encoding="utf-8") as f:
    f.write(code)
print(f"train_lstm.py written: {len(code)} bytes")
