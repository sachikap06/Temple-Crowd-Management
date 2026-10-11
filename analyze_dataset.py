"""
analyze_dataset.py – Data Quality Inspection Script
Smart Temple Crowd Management Project
"""
import pandas as pd
import numpy as np

df = pd.read_csv("output/crowd_dataset.csv")

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)
print(f"Total observations: {len(df)}")
print(f"Columns: {list(df.columns)}")

videos = sorted(df["Video_Name"].unique().tolist())
print(f"Videos ({len(videos)}): {videos}")

print()
print("=== PER-VIDEO STATISTICS ===")
per_video = df.groupby("Video_Name")["People_Count"].agg(["count", "min", "max", "mean"]).round(1)
print(per_video.to_string())

counts = df["People_Count"]
print()
print("=== GLOBAL STATS ===")
print(f"Min:    {counts.min()}")
print(f"Max:    {counts.max()}")
print(f"Mean:   {counts.mean():.2f}")
print(f"Median: {counts.median():.2f}")
print(f"Std:    {counts.std():.2f}")
print(f"P25:    {counts.quantile(0.25):.2f}")
print(f"P50:    {counts.quantile(0.50):.2f}")
print(f"P75:    {counts.quantile(0.75):.2f}")
print(f"P90:    {counts.quantile(0.90):.2f}")
print(f"P95:    {counts.quantile(0.95):.2f}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate (video+time): {df.duplicated(subset=['Video_Name','Time_Seconds']).sum()}")

print()
print("=== STATUS DISTRIBUTION ===")
print(df["Crowd_Status"].value_counts())

print()
print("=== THRESHOLD PER VIDEO ===")
thresh_per_video = df.groupby("Video_Name")["Threshold"].first()
print(thresh_per_video.to_string())

# Sudden changes analysis
df_sorted = df.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)
df_sorted["count_diff"] = df_sorted.groupby("Video_Name")["People_Count"].diff()

print()
print("=== SUDDEN CHANGES ===")
print(f"Max sudden increase: {df_sorted['count_diff'].max()}")
print(f"Max sudden decrease: {df_sorted['count_diff'].min()}")
print(f"Changes > 20 people in one step: {(df_sorted['count_diff'].abs() > 20).sum()}")

suspicious = df_sorted[df_sorted["count_diff"].abs() > 20][
    ["Video_Name", "Time_Seconds", "People_Count", "count_diff"]
]
print("Suspicious jumps (>20 count change):")
print(suspicious.to_string())

print()
print("=== NORMALIZATION CHECK ===")
# What the training code would see
# 70% of each video's data goes to training
train_maxes = []
for vname in videos:
    v_df = df_sorted[df_sorted["Video_Name"] == vname].reset_index(drop=True)
    counts_v = v_df["People_Count"].values
    n_train = max(1, int(0.70 * len(counts_v)))
    train_maxes.append(counts_v[:n_train].max())
    print(f"  {vname}: total={len(counts_v)}, train_n={n_train}, train_max={counts_v[:n_train].max()}")

overall_train_max = max(train_maxes)
print(f"Overall train max (normalization scale): {overall_train_max}")
print(f"Dataset overall max (for reference): {df['People_Count'].max()}")
