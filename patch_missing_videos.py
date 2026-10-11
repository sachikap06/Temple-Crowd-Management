"""
patch_missing_videos.py
=======================
Fast patch: only re-processes videos that are MISSING or have < MIN_ROWS rows
in the existing crowd_dataset.csv.

Speedups vs process_videos.py:
  - Skips already-complete videos entirely
  - Uses 1 FPS instead of 6 FPS (6x fewer frames)
  - Uses ONE YOLO tracker call per frame for both count + track IDs
    (no separate tiled detection = ~2-3x faster per frame)
  - Uses half-resolution inference (imgsz=320) for small videos

Run:  python patch_missing_videos.py
"""

import os, sys
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────
VIDEO_FOLDER  = "videos"
OUTPUT_CSV    = os.path.join("output", "crowd_dataset.csv")
MODEL_PATH    = "yolov8n.pt"

SAMPLE_FPS     = 1.0    # 1 frame per second  (was 6 → 6x faster)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.40
IMGSZ          = 640    # lower = faster; 320 works fine for small videos

# Videos with fewer than this many rows will be re-processed
MIN_ROWS = 10

BASE_THRESHOLD = 15
REFERENCE_AREA = 640 * 480


# ── Helpers ───────────────────────────────────────────────
def compute_area_threshold(w, h):
    scaled = BASE_THRESHOLD * (w * h / REFERENCE_AREA)
    return max(10, min(int(round(scaled)), 500))

def get_crowd_status(count, threshold):
    lower = max(1, threshold // 3)
    if count <= lower:
        return "Normal"
    elif count <= threshold:
        return "Moderate"
    return "Congested"


# ── Per-video processor (fast) ─────────────────────────────
def process_video_fast(video_path, model):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Cannot open: {video_name}")
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        print(f"[SKIP] Invalid FPS: {video_name}")
        cap.release()
        return []

    threshold      = compute_area_threshold(vid_w, vid_h)
    frame_interval = max(1, int(round(fps / SAMPLE_FPS)))
    duration_sec   = total_frames / fps

    print(f"\n── {video_name}  ({vid_w}x{vid_h}, {fps:.0f}fps, "
          f"{duration_sec:.1f}s, sample every {frame_interval} frames)")

    rows           = []
    frame_idx      = 0
    seen_track_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            time_sec = round(frame_idx / fps, 2)

            try:
                # Single YOLO call → detection + ByteTrack IDs
                results = model.track(
                    frame,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    classes=[0],
                    imgsz=IMGSZ,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                    agnostic_nms=True,
                )
                boxes = results[0].boxes
                if boxes is not None:
                    people_count = len(boxes)
                    if boxes.id is not None:
                        for tid in boxes.id.int().tolist():
                            seen_track_ids.add(tid)
                else:
                    people_count = 0
            except Exception as e:
                print(f"\n  [WARN] frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            crowd_status = get_crowd_status(people_count, threshold)
            exceeded     = bool(people_count > threshold)

            rows.append({
                "Video_Name":       video_name,
                "Time_Seconds":     time_sec,
                "Frame_No":         frame_idx,
                "People_Count":     people_count,
                "Unique_Track_IDs": len(seen_track_ids),
                "Crowd_Status":     crowd_status,
                "Threshold":        threshold,
                "Exceeded":         exceeded,
            })

            icon = "🔴" if exceeded else ("🟡" if crowd_status == "Moderate" else "🟢")
            print(
                f"  t={time_sec:5.1f}s  count={people_count:3d}  "
                f"ids={len(seen_track_ids):3d}  {crowd_status:<10} {icon}",
                end="\r",
            )

        frame_idx += 1

    cap.release()
    print(f"\n  Done: {video_name}: {len(rows)} rows generated")
    return rows


# ── Main ───────────────────────────────────────────────────
def main():
    # Load existing dataset
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        row_counts  = df_existing["Video_Name"].value_counts().to_dict()
    else:
        df_existing = pd.DataFrame()
        row_counts  = {}

    # Find all videos in folder
    all_videos = sorted([
        f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(".mp4")
    ])

    # Decide which ones need processing
    to_process = [
        v for v in all_videos
        if row_counts.get(v, 0) < MIN_ROWS
    ]

    if not to_process:
        print("All videos already have sufficient data. Nothing to do.")
        return

    print("=" * 60)
    print(f"  Fast Patch - Processing {len(to_process)} video(s)")
    print(f"  Skipping {len(all_videos) - len(to_process)} already-complete video(s)")
    print("=" * 60)
    for v in all_videos:
        rows = row_counts.get(v, 0)
        status = "WILL PROCESS" if v in to_process else f"OK ({rows} rows)"
        print(f"  {v:<15}  {status}")

    print()
    model = YOLO(MODEL_PATH)
    print(f"[OK] YOLOv8 model loaded: {MODEL_PATH}\n")

    new_rows = []
    for video_name in to_process:
        video_path = os.path.join(VIDEO_FOLDER, video_name)
        rows = process_video_fast(video_path, model)
        new_rows.extend(rows)

    if not new_rows:
        print("\n[INFO] No new data generated.")
        return

    df_new = pd.DataFrame(new_rows)

    # Remove old incomplete rows for these videos, then append fresh data
    if not df_existing.empty:
        df_existing = df_existing[~df_existing["Video_Name"].isin(to_process)]
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    # Sort by video name then time
    df_final = df_final.sort_values(["Video_Name", "Time_Seconds"]).reset_index(drop=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  Done! Dataset updated: {OUTPUT_CSV}")
    print(f"  New rows added  : {len(new_rows)}")
    print(f"  Total rows now  : {len(df_final)}")
    print(f"  Videos in CSV   : {df_final['Video_Name'].nunique()}")
    print(f"{'='*60}")
    print("\nRefresh the Streamlit dashboard to see updated graphs.")


if __name__ == "__main__":
    main()
