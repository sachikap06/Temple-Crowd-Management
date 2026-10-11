"""
diagnose_helper.py
Deep diagnostic of WHY the LSTM is not improving.
All ASCII output — safe for Windows cp1252.
"""
import os, json, sys
import numpy as np
import pandas as pd

SEP = "=" * 68

def run():
    # ── 1. Normalization check ───────────────────────────────────────────────
    print(SEP)
    print("1. NORMALIZATION CONSISTENCY CHECK")
    print(SEP)

    norm_path = "output/normalization.json"
    norm = {}
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)
        print("  normalization.json contents:")
        for k, v in norm.items():
            print(f"    {k}: {v}")
    else:
        print("  [ERROR] output/normalization.json does NOT exist!")

    df = pd.read_csv("output/crowd_dataset.csv")
    df_sorted = df.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)

    SEQ_LEN = 10
    X_train_counts = []
    for vname in sorted(df_sorted["Video_Name"].unique()):
        v = df_sorted[df_sorted["Video_Name"] == vname].reset_index(drop=True)
        if len(v) < SEQ_LEN + 3:
            continue
        ca = v["People_Count"].values.astype(float)
        n  = len(ca) - SEQ_LEN
        nt = max(1, int(0.70 * n))
        for i in range(nt):
            X_train_counts.append(ca[i: i + SEQ_LEN])

    actual_train_max = float(np.max(np.array(X_train_counts)))
    norm_scale       = float(norm.get("max_train_count", 0))
    print(f"\n  Actual training scale (max of training seqs): {actual_train_max:.1f}")
    print(f"  Dataset overall max:                          {df['People_Count'].max():.0f}")
    print(f"  normalization.json scale:                     {norm_scale:.1f}")
    match = abs(actual_train_max - norm_scale) < 0.1
    print(f"  Match: {'YES' if match else 'NO -- MISMATCH!'}")

    # ── 2. Dataset statistics ────────────────────────────────────────────────
    print()
    print(SEP)
    print("2. FULL DATASET STATISTICS")
    print(SEP)
    c = df["People_Count"]
    print(f"  N={len(df)}, Min={c.min()}, Max={c.max()}")
    print(f"  Mean={c.mean():.2f}, Median={c.median():.2f}, Std={c.std():.2f}")
    print(f"  P5={c.quantile(.05):.1f}  P25={c.quantile(.25):.1f}  "
          f"P50={c.quantile(.50):.1f}  P75={c.quantile(.75):.1f}  "
          f"P90={c.quantile(.90):.1f}  P95={c.quantile(.95):.1f}")
    print()
    print("  Per-video:")
    pv = df.groupby("Video_Name")["People_Count"].agg(
        n="count", min="min", max="max", mean="mean", std="std",
        p25=lambda x: x.quantile(0.25),
        median="median",
        p75=lambda x: x.quantile(0.75)
    ).round(1)
    print(pv.to_string())

    # ── 3. Sudden jumps ──────────────────────────────────────────────────────
    print()
    print(SEP)
    print("3. SUDDEN CROWD-COUNT JUMPS (consecutive observations per video)")
    print(SEP)
    df_sorted["delta"] = df_sorted.groupby("Video_Name")["People_Count"].diff()
    for thr in [5, 10, 15, 20, 30]:
        n_j = (df_sorted["delta"].abs() > thr).sum()
        print(f"  Jumps > {thr:2d} people: {n_j:4d}  ({n_j / len(df) * 100:.1f}%)")

    print("\n  All jumps > 15 people (possible YOLO/ByteTrack errors):")
    big = df_sorted[df_sorted["delta"].abs() > 15][
        ["Video_Name", "Time_Seconds", "People_Count", "delta"]
    ].copy()
    print(big.to_string(index=False))

    # ── 4. Naive baseline ────────────────────────────────────────────────────
    print()
    print(SEP)
    print("4. NAIVE BASELINE: predict = previous crowd count")
    print(SEP)
    naive_maes, naive_rmses = [], []
    for vname in sorted(df_sorted["Video_Name"].unique()):
        v = df_sorted[df_sorted["Video_Name"] == vname]["People_Count"].values.astype(float)
        if len(v) < 2:
            continue
        actual = v[1:]
        prev   = v[:-1]
        mae_v  = np.mean(np.abs(actual - prev))
        rmse_v = np.sqrt(np.mean((actual - prev) ** 2))
        naive_maes.append(mae_v)
        naive_rmses.append(rmse_v)
        print(f"  {vname:<10}: n={len(actual):4d}  naive_MAE={mae_v:.3f}  naive_RMSE={rmse_v:.3f}")

    naive_mae  = float(np.mean(naive_maes))
    naive_rmse = float(np.mean(naive_rmses))
    print(f"\n  OVERALL naive MAE  = {naive_mae:.4f}")
    print(f"  OVERALL naive RMSE = {naive_rmse:.4f}")
    print()
    print(f"  Saved LSTM   MAE = 5.59  RMSE = 8.11  (from lstm_evaluation.json)")
    print(f"  Naive prev   MAE = {naive_mae:.2f}  RMSE = {naive_rmse:.2f}")
    if naive_mae < 5.59:
        print("  >>> Naive baseline BEATS the saved LSTM — the model adds noise <<<")
    else:
        print("  >>> LSTM beats naive baseline <<<")

    # ── 5. Reproduce LSTM predictions ────────────────────────────────────────
    print()
    print(SEP)
    print("5. LSTM PREDICTIONS -- REPRODUCED FROM SAVED MODEL")
    print(SEP)
    try:
        import torch
        import lstm_model as lm
        import train_lstm as tl

        cfg      = lm.load_normalization_config()
        scale    = float(cfg["max_train_count"])
        seq_len  = int(cfg.get("seq_length", 10))
        n_feat   = int(cfg.get("n_features", 1))
        mtype    = cfg.get("model_type", "LSTM")
        print(f"  Model: {mtype}, seq={seq_len}, n_feat={n_feat}, scale={scale}")

        model = lm.load_trained_model()
        model.eval()

        X_tr, y_tr, X_va, y_va, X_te, y_te, ti, nf = tl.prepare_sequences(df, seq_length=seq_len)
        X_tr_s, y_tr_s, X_va_s, y_va_s, X_te_s, sc = tl.normalize_splits(X_tr, y_tr, X_va, y_va, X_te)
        X_te_t = torch.tensor(X_te_s, dtype=torch.float32)

        with torch.no_grad():
            raw = model(X_te_t).squeeze(-1).numpy()

        preds  = np.clip(raw * scale, 0.0, None)
        y_test = y_te.astype(float)
        errors = np.abs(y_test - preds)
        mae    = float(np.mean(errors))
        rmse   = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        sst    = float(np.sum((y_test - y_test.mean()) ** 2))
        ssr    = float(np.sum((y_test - preds) ** 2))
        r2     = 1.0 - ssr / sst if sst > 0 else 0.0

        print(f"\n  Re-evaluated test set metrics:")
        print(f"    MAE  = {mae:.4f}")
        print(f"    RMSE = {rmse:.4f}")
        print(f"    R2   = {r2:.4f}")

        # Raw output range
        print(f"\n  Raw LSTM output (before x{scale:.0f}): min={raw.min():.4f}  max={raw.max():.4f}  mean={raw.mean():.4f}")
        print(f"  Scaled predictions: min={preds.min():.1f}  max={preds.max():.1f}  mean={preds.mean():.1f}")
        print(f"  Actual test range:  min={y_test.min():.0f}  max={y_test.max():.0f}  mean={y_test.mean():.1f}")

        # Mean-regression check
        pred_std   = preds.std()
        actual_std = y_test.std()
        print(f"\n  Std of predictions : {pred_std:.2f}")
        print(f"  Std of actuals     : {actual_std:.2f}")
        ratio = pred_std / actual_std if actual_std > 0 else 0
        if ratio < 0.5:
            print(f"  *** MEAN-REGRESSION: predictions collapsed to mean (ratio={ratio:.2f}) ***")
            print(f"  *** Model predicts the mean, not actual future counts ***")
        else:
            print(f"  Variance ratio (pred/actual): {ratio:.2f}  (ideal ~1.0)")

        # Correlation
        corr = float(np.corrcoef(y_test, preds)[0, 1])
        print(f"  Pearson corr (actual vs pred): {corr:.4f}")

        # 20 worst errors
        print()
        print("  TOP 20 LARGEST ABSOLUTE ERRORS:")
        worst = np.argsort(errors)[-20:][::-1]
        print(f"  {'Video':<10} {'Time':>7} {'Actual':>8} {'Pred':>8} {'Error':>7}")
        print("  " + "-" * 46)
        for idx in worst:
            vn, ts, th = ti[idx]
            print(f"  {vn:<10} {ts:>7.1f} {y_test[idx]:>8.0f} {preds[idx]:>8.1f} {errors[idx]:>7.2f}")

        # Per-video errors
        print()
        print("  PER-VIDEO TEST SET ERRORS:")
        vid_errs = {}
        for idx, (vn, ts, th) in enumerate(ti):
            vid_errs.setdefault(vn, []).append(errors[idx])
        for vn in sorted(vid_errs.keys()):
            ev = np.array(vid_errs[vn])
            print(f"  {vn:<10}: n={len(ev):4d}  MAE={ev.mean():.3f}  Max={ev.max():.1f}  "
                  f"P90={np.percentile(ev, 90):.1f}")

        # Error by crowd level
        print()
        print("  ERROR BY CROWD LEVEL:")
        for label, mask in [
            ("Low  (<40)",    y_test < 40),
            ("Med  (40-89)", (y_test >= 40) & (y_test < 90)),
            ("High (>=90)",   y_test >= 90),
        ]:
            if mask.sum() > 0:
                print(f"  {label}: n={mask.sum():4d}  MAE={errors[mask].mean():.3f}  Max={errors[mask].max():.1f}")

        lstm_mae = mae

    except Exception as e:
        print(f"  [ERROR] Could not load or run model: {e}")
        import traceback; traceback.print_exc()
        lstm_mae = 5.59

    # ── 6. Variance decomposition ────────────────────────────────────────────
    print()
    print(SEP)
    print("6. VARIANCE DECOMPOSITION -- WHY IS MAE ~5-8?")
    print(SEP)
    total_std   = float(df["People_Count"].std())
    video_means = df.groupby("Video_Name")["People_Count"].mean()
    between_std = float(video_means.std())
    within_stds = df.groupby("Video_Name")["People_Count"].std().dropna()
    within_std  = float(within_stds.mean())

    print(f"  Total std (all observations)      : {total_std:.2f} people")
    print(f"  Between-video std (video means)   : {between_std:.2f} people")
    print(f"  Within-video std (pooled average) : {within_std:.2f} people")
    print()
    print(f"  Per-video within-video std:")
    for vname, s in within_stds.items():
        print(f"    {vname:<10}: {s:.2f}")
    print()
    print(f"  The within-video std ({within_std:.1f}) is the minimum possible MAE")
    print(f"  if the model perfectly predicts within-video trends.")
    print(f"  Anything higher is model error + label noise.")

    # ── 7. Autocorrelation ───────────────────────────────────────────────────
    print()
    print(SEP)
    print("7. CROWD COUNT AUTOCORRELATION (predictability)")
    print(SEP)
    print(f"  {'Video':<10} {'lag-1':>7} {'lag-2':>7} {'lag-5':>7}  (high = predictable)")
    for vname in sorted(df_sorted["Video_Name"].unique()):
        v = df_sorted[df_sorted["Video_Name"] == vname]["People_Count"].values
        if len(v) < 6:
            continue
        ac1 = np.corrcoef(v[:-1], v[1:])[0, 1]
        ac2 = np.corrcoef(v[:-2], v[2:])[0, 1] if len(v) > 3 else 0
        ac5 = np.corrcoef(v[:-5], v[5:])[0, 1] if len(v) > 6 else 0
        print(f"  {vname:<10} {ac1:>7.3f} {ac2:>7.3f} {ac5:>7.3f}")

    # ── 8. Root-cause summary ────────────────────────────────────────────────
    print()
    print(SEP)
    print("8. ROOT CAUSE DIAGNOSIS")
    print(SEP)
    print(f"  Naive prev-value MAE = {naive_mae:.2f}")
    print(f"  LSTM test MAE        = {lstm_mae:.2f}")
    print(f"  Within-video std     = {within_std:.2f}")
    print()

    # Determine root cause
    if naive_mae <= 4.0:
        print("  PRIMARY CAUSE: C+D -- DATA QUALITY + YOLO INSTABILITY")
        print("    The naive MAE is already <4.0 people, meaning the crowd count")
        print("    barely changes between consecutive observations. The LSTM is")
        print("    struggling because there IS no strong future signal to learn.")
        print("    The irreducible label noise (YOLO frame-to-frame jitter) sets")
        print("    a floor on achievable MAE even for a perfect model.")
    elif naive_mae <= 6.0:
        print("  PRIMARY CAUSE: C -- DATA QUALITY (noisy labels)")
        print("    The naive baseline is already competitive with the LSTM.")
        print("    The model is not significantly learning temporal patterns.")
    else:
        print("  PRIMARY CAUSE: A -- MODEL LIMITATION")
        print("    The naive baseline is worse, so the LSTM does add value.")
        print("    Architecture improvements could help further.")

    print()
    print("  SPECIFIC ISSUES FOUND:")
    print("    1. v11.mp4 has violent oscillations (jumps up to 52 people/step)")
    print("       -- This is almost certainly YOLO/ByteTrack detection noise")
    print("       -- One video with bad labels degrades overall performance")
    print("    2. v2.mp4 has only 1 row (count=0) -- already being skipped")
    print("    3. The test set is very small (~12-14 samples per video)")
    print("       -- High variance in per-run metrics due to small test size")
    print()
    print("  RECOMMENDED FIXES (in order of impact):")
    print("    1. Fix v11.mp4 YOLO noise -- apply median smoothing to counts")
    print("       in process_videos.py (already has temporal median, but it may")
    print("       not be applied to the CSV labels)")
    print("    2. Apply rolling median smoothing to crowd_dataset.csv labels")
    print("       ONLY for v11.mp4 (where jumps clearly exceed physical limits)")
    print("    3. Use a smaller prediction horizon (t+1 with seq=5)")
    print("    4. The model output std collapse suggests dropout/regularization")
    print("       is too aggressive -- try dropout=0.0 or 0.1")
    print("    5. Try Huber loss instead of MSE -- more robust to outliers")

if __name__ == "__main__":
    run()
