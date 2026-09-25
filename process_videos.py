"""
process_videos.py
=================
Multi-Video Crowd Analysis and Alert System
Smart Temple Crowd Management Project

Features:
  - Scans 'videos/' folder for all .mp4 files automatically
  - TILED inference: splits each frame into overlapping 640x640 tiles for
    accurate detection in ultra-dense crowds (200-400+ people per frame)
  - AREA-BASED threshold: auto-scales alert threshold to the visible area of
    each video (larger frame resolution = larger crowd zone = higher threshold)
  - Saves results to output/crowd_dataset.csv with columns:
      Video_Name, Time_Seconds, People_Count, Crowd_Status, Threshold, Exceeded

Usage:
    python process_videos.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_FOLDER  = "videos"
OUTPUT_FOLDER = "output"
OUTPUT_CSV    = os.path.join(OUTPUT_FOLDER, "crowd_dataset.csv")
MODEL_PATH    = "yolov8n.pt"

# YOLO settings (tuned for ultra-dense crowd scenes)
CONF_THRESHOLD = 0.18    # Very low conf to catch occluded/distant people
IOU_THRESHOLD  = 0.30    # Low IoU allows tightly packed boxes (crowd)
TILE_SIZE      = 640     # Each tile is processed at this resolution
TILE_OVERLAP   = 0.30    # 30% overlap prevents missing people at tile edges

SAMPLE_FPS     = 1       # Sample ~1 frame per second

# Area-based threshold calibration
# Reference: 640×480 resolution (307,200 pixels) → threshold = 15
BASE_THRESHOLD = 15
REFERENCE_AREA = 640 * 480   # = 307,200 pixels


# ==========================================
# AREA-BASED THRESHOLD
# ==========================================

def compute_area_threshold(frame_width: int, frame_height: int) -> int:
    """
    Auto-scale the crowd alert threshold to the video's visible frame area.

    Logic: A larger frame captures a larger physical space. More people are
    expected in a larger area before the crowd density becomes dangerous.
    This prevents a large hall being flagged at just 15 people, or a small
    gate view being set too high.

    Calibration table:
      Resolution    | Area         | Threshold
      640  × 480    | 307,200 px   |  15
      1280 × 720    | 921,600 px   |  45
      1920 × 1080   | 2,073,600 px | 101
      2560 × 1440   | 3,686,400 px | 180
    """
    frame_area = frame_width * frame_height
    scaled     = BASE_THRESHOLD * (frame_area / REFERENCE_AREA)
    return max(10, min(int(round(scaled)), 500))


# ==========================================
# CROWD STATUS & ALERT
# ==========================================

def get_crowd_status(count: int, threshold: int) -> str:
    """
    Return crowd status label relative to the area-based threshold.
    Normal boundary = 1/3 of threshold; Moderate = up to threshold.
    """
    lower_bound = max(1, threshold // 3)
    if count <= lower_bound:
        return "Normal"
    elif count <= threshold:
        return "Moderate"
    else:
        return "Congested"


def check_alert(count: int, threshold: int, video_name: str, time_sec: float) -> None:
    """Print a visible terminal alert when people count exceeds threshold."""
    print(
        f"\n{'='*62}\n"
        f"  ALERT: Crowd has exceeded the threshold!\n"
        f"  Video       : {video_name}\n"
        f"  Time        : {time_sec:.1f}s\n"
        f"  Detected    : {count} people\n"
        f"  Threshold   : {threshold} people  (area-based auto-scaled)\n"
        f"  Exceeded by : +{count - threshold} people above limit\n"
        f"{'='*62}\n"
    )


# ==========================================
# LOAD YOLO MODEL
# ==========================================

def load_yolo_model(model_path: str) -> YOLO:
    """Load YOLOv8 model with clear error handling."""
    if not os.path.exists(model_path):
        print(f"[ERROR] YOLO model file not found: '{model_path}'")
        print("        Ensure 'yolov8n.pt' is in the project root directory.")
        sys.exit(1)
    try:
        model = YOLO(model_path)
        print(f"[OK] YOLOv8 model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)


# ==========================================
# TILED DETECTION FOR DENSE CROWDS
# ==========================================

def detect_with_tiling(model: YOLO, frame: np.ndarray) -> int:
    """
    Slice the frame into overlapping tiles, run YOLO person detection on
    each tile, translate tile-local coordinates back to full-frame space,
    then apply global NMS to remove cross-tile duplicate detections.

    Why this works for ultra-dense crowds:
    - Running YOLO on the full downsized image causes people to appear tiny
      → model misses most of them.
    - Processing each 640×640 tile at full resolution dramatically increases
      the apparent person size → many more accurate detections.
    - Overlapping tiles ensure nobody near tile borders is missed.
    - Global NMS removes the same person counted in two adjacent tiles.

    Returns:
        int: Unique people count in the full frame.
    """
    h, w   = frame.shape[:2]
    stride = int(TILE_SIZE * (1 - TILE_OVERLAP))

    all_boxes  = []   # [x1, y1, x2, y2] in full-frame coords
    all_scores = []

    for y_start in range(0, h, stride):
        for x_start in range(0, w, stride):
            x1 = x_start
            y1 = y_start
            x2 = min(x_start + TILE_SIZE, w)
            y2 = min(y_start + TILE_SIZE, h)

            # Skip tiny edge tiles
            if (x2 - x1) < 64 or (y2 - y1) < 64:
                continue

            tile = frame[y1:y2, x1:x2]

            try:
                results = model(
                    tile,
                    conf=CONF_THRESHOLD,
                    classes=[0],          # 0 = person (COCO)
                    imgsz=TILE_SIZE,
                    iou=IOU_THRESHOLD,
                    agnostic_nms=True,    # Better for overlapping crowd
                    verbose=False,
                )
            except Exception:
                continue

            if results[0].boxes is None:
                continue

            for box in results[0].boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                # Map tile-local coords → full-frame coords
                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                all_scores.append(float(box.conf[0]))

    if not all_boxes:
        return 0

    # Global NMS to deduplicate cross-tile detections
    boxes_xywh = [
        [b[0], b[1], b[2] - b[0], b[3] - b[1]]
        for b in all_boxes
    ]
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh, all_scores, CONF_THRESHOLD, IOU_THRESHOLD
    )
    return int(len(indices)) if len(indices) > 0 else 0


# ==========================================
# PROCESS A SINGLE VIDEO
# ==========================================

def process_video(video_path: str, model: YOLO) -> list[dict]:
    """
    Process one video file:
      - Reads frame dimensions to compute area-based threshold
      - Samples ~1 frame per second
      - Runs tiled YOLO detection on each sampled frame
      - Triggers alerts when count exceeds the area threshold
      - Returns list of row dicts for crowd_dataset.csv
    """
    video_name = os.path.basename(video_path)
    print(f"\n{'─'*62}")
    print(f"[INFO] Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: '{video_path}'. Skipping.")
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        print(f"[WARNING] Invalid FPS for '{video_name}'. Skipping.")
        cap.release()
        return []

    area_threshold = compute_area_threshold(vid_w, vid_h)
    duration_sec   = total_frames / fps
    frame_interval = max(1, int(round(fps / SAMPLE_FPS)))

    print(f"         Resolution  : {vid_w} x {vid_h}")
    print(f"         FPS/Frames  : {fps:.1f} / {total_frames}")
    print(f"         Duration    : {duration_sec:.1f}s")
    print(f"         Sample every: {frame_interval} frame(s)")
    print(f"         Threshold   : {area_threshold} people  "
          f"(auto-scaled for {vid_w}x{vid_h} frame area)")
    print(f"{'─'*62}")

    rows        = []
    frame_idx   = 0
    sample_num  = 0
    alert_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            time_sec = round(frame_idx / fps, 2)

            if frame is None or frame.size == 0:
                frame_idx += 1
                continue

            try:
                people_count = detect_with_tiling(model, frame)
            except Exception as e:
                print(f"\n[WARNING] Detection error at frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            crowd_status = get_crowd_status(people_count, area_threshold)
            exceeded     = bool(people_count > area_threshold)

            if exceeded:
                alert_count += 1
                check_alert(people_count, area_threshold, video_name, time_sec)

            rows.append({
                "Video_Name":   video_name,
                "Time_Seconds": time_sec,
                "People_Count": people_count,
                "Crowd_Status": crowd_status,
                "Threshold":    area_threshold,
                "Exceeded":     exceeded,
            })

            sample_num += 1
            icon = "🔴" if exceeded else ("🟡" if crowd_status == "Moderate" else "🟢")
            print(
                f"  t={time_sec:6.1f}s  Count={people_count:4d}  "
                f"Threshold={area_threshold}  {crowd_status:<10} {icon}",
                end="\r",
            )

        frame_idx += 1

    cap.release()
    print(
        f"\n[OK] '{video_name}': "
        f"{sample_num} samples | {alert_count} alert(s) | "
        f"Threshold={area_threshold}"
    )
    return rows


# ==========================================
# PROCESS ALL VIDEOS IN FOLDER
# ==========================================

def process_all_videos(video_folder: str, model: YOLO) -> pd.DataFrame:
    """Scan video_folder for all .mp4 files and process each independently."""
    if not os.path.isdir(video_folder):
        print(f"[ERROR] Video folder not found: '{video_folder}'")
        print("        Create a 'videos/' folder and place .mp4 files inside.")
        sys.exit(1)

    mp4_files = sorted([
        f for f in os.listdir(video_folder)
        if f.lower().endswith(".mp4")
    ])

    if not mp4_files:
        print(
            f"[ERROR] No .mp4 files found in '{video_folder}'.\n"
            "        Copy your .mp4 videos into the 'videos/' folder and retry."
        )
        sys.exit(1)

    print(f"\n[INFO] Found {len(mp4_files)} video(s): {mp4_files}")

    all_rows = []
    for filename in mp4_files:
        rows = process_video(os.path.join(video_folder, filename), model)
        all_rows.extend(rows)

    if not all_rows:
        print("[WARNING] No valid data collected from any video.")
        return pd.DataFrame(columns=[
            "Video_Name", "Time_Seconds", "People_Count",
            "Crowd_Status", "Threshold", "Exceeded",
        ])

    return pd.DataFrame(all_rows)


# ==========================================
# SAVE RESULTS
# ==========================================

def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save crowd dataset to CSV with all columns."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cols = ["Video_Name", "Time_Seconds", "People_Count",
            "Crowd_Status", "Threshold", "Exceeded"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(output_path, index=False)

    print(f"\n[OK] Dataset saved: {output_path}")
    print(f"     Total rows : {len(df)}")
    print(f"     Videos     : {df['Video_Name'].nunique()}")
    if "Exceeded" in df.columns:
        print(f"     Alerts     : {int(df['Exceeded'].sum())} threshold-exceeded frames")


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 62)
    print("  Smart Temple – Multi-Video Crowd Analysis System")
    print("  Mode: Tiled Inference + Area-Based Adaptive Threshold")
    print("=" * 62)

    model = load_yolo_model(MODEL_PATH)
    df    = process_all_videos(VIDEO_FOLDER, model)

    if df.empty:
        print("[INFO] No data to save. Exiting.")
        return

    save_dataset(df, OUTPUT_CSV)

    print("\n--- Per-Video Summary ---")
    summary = df.groupby("Video_Name").agg(
        Samples   =("People_Count", "count"),
        Avg_Count =("People_Count", "mean"),
        Max_Count =("People_Count", "max"),
        Threshold =("Threshold",    "first"),
        Alerts    =("Exceeded",     "sum"),
    ).round({"Avg_Count": 1})
    print(summary.to_string())
    print("\n[DONE] All videos processed successfully.")
    print("       Run: streamlit run dashboard.py\n")


if __name__ == "__main__":
    main()
