"""
fix_labels.py
=============
Apply conservative rolling-median smoothing to v11.mp4 ONLY.

JUSTIFICATION:
  v11.mp4 has naive_MAE = 11.5 people, with jumps of 20-55 people in
  a single 0.2-second step at 5fps. At 5fps, a real crowd cannot change
  by 50 people in 0.2 seconds (that would require ~250 people/second flow).
  These are YOLO/ByteTrack detection artifacts (occlusion, re-ID failure),
  NOT genuine crowd changes.

  A window-5 rolling median preserves real trends while suppressing
  transient detection spikes. We do NOT apply this to other videos
  where within-video std is plausible.

  We NEVER:
    - Remove rows
    - Add rows
    - Touch the test set after splitting
    - Smooth labels for videos that look fine

RESULT: The smoothed CSV replaces crowd_dataset.csv.
  The original is backed up as crowd_dataset_original.csv.
"""

import pandas as pd
import numpy as np
import os, shutil

DATASET_CSV  = "output/crowd_dataset.csv"
BACKUP_CSV   = "output/crowd_dataset_original.csv"

# Videos to smooth and their window sizes
# Only v11.mp4 has physically-impossible jumps (naive_MAE=11.5, jumps>20)
SMOOTH_VIDEOS = {
    "v11.mp4": 5,   # window=5 median  (~1 second at 5fps)
}

df = pd.read_csv(DATASET_CSV)

# Backup original if not already backed up
if not os.path.exists(BACKUP_CSV):
    shutil.copy(DATASET_CSV, BACKUP_CSV)
    print(f"Backed up original to {BACKUP_CSV}")

print("Before smoothing:")
for vname, window in SMOOTH_VIDEOS.items():
    v = df[df["Video_Name"] == vname]["People_Count"]
    naive_mae = float(np.mean(np.abs(v.diff().dropna().abs())))
    print(f"  {vname}: n={len(v)}, std={v.std():.2f}, naive_MAE(step)={naive_mae:.2f}")

# Apply smoothing per-video
df_sorted = df.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)

for vname, window in SMOOTH_VIDEOS.items():
    mask = df_sorted["Video_Name"] == vname
    raw  = df_sorted.loc[mask, "People_Count"].values.astype(float)
    smoothed = pd.Series(raw).rolling(window=window, center=True, min_periods=1).median().round().astype(int).values
    df_sorted.loc[mask, "People_Count"] = smoothed
    # Recompute Crowd_Status and Exceeded from smoothed counts
    thresh = df_sorted.loc[mask, "Threshold"].iloc[0]
    low    = max(1, thresh // 3)
    statuses = []
    exceededs = []
    for cnt in smoothed:
        if cnt <= low:
            statuses.append("Normal")
        elif cnt <= thresh:
            statuses.append("Moderate")
        else:
            statuses.append("Congested")
        exceededs.append(bool(cnt > thresh))
    df_sorted.loc[mask, "Crowd_Status"] = statuses
    df_sorted.loc[mask, "Exceeded"]     = exceededs

print("\nAfter smoothing:")
for vname, window in SMOOTH_VIDEOS.items():
    v = df_sorted[df_sorted["Video_Name"] == vname]["People_Count"]
    naive_mae = float(np.mean(np.abs(v.diff().dropna().abs())))
    print(f"  {vname}: n={len(v)}, std={v.std():.2f}, naive_MAE(step)={naive_mae:.2f}")

# Verify: check largest jumps remaining
df_sorted["delta"] = df_sorted.groupby("Video_Name")["People_Count"].diff()
print("\nJumps > 15 after smoothing:")
big = df_sorted[df_sorted["delta"].abs() > 15][
    ["Video_Name", "Time_Seconds", "People_Count", "delta"]
]
if big.empty:
    print("  None -- smoothing resolved all large spikes")
else:
    print(big.to_string(index=False))
    print("  (remaining jumps are from other videos and may be genuine)")

# Save
df_out = df_sorted.drop(columns=["delta"])
df_out.to_csv(DATASET_CSV, index=False)
print(f"\nSaved smoothed dataset to {DATASET_CSV}")
print(f"Total rows: {len(df_out)}  (unchanged -- no rows added or removed)")
