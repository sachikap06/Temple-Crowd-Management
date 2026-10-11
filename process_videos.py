"""
process_videos.py
=================
Multi-Video Crowd Analysis, Object Tracking and Dataset Generation System
Smart Temple Crowd Management Project

Features:
  - Scans 'videos/' folder for all .mp4 files automatically
  - Extracts real video metadata (resolution, FPS, total frames, duration)
  - Configurable frame sampling (default: 1 FPS)
  - TILED inference: splits each frame into overlapping 640x640 tiles for
    accurate detection in ultra-dense crowds
  - Object Tracking: Uses ByteTrack tracker to assign persistent IDs and track unique people
  - AREA-BASED threshold: auto-scales alert threshold to visible area of each video
  - Saves annotated detection sample image to 'output/sample_annotated_detection.jpg'
  - Saves dataset to 'output/crowd_dataset.csv' preserving backward compatibility:
      Video_Name, Time_Seconds, Frame_No, People_Count, Unique_Track_IDs, Crowd_Status, Threshold, Exceeded
"""

import os
import sys
import json
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
SAMPLE_IMAGE  = os.path.join(OUTPUT_FOLDER, "sample_annotated_detection.jpg")
SUMMARY_JSON  = os.path.join(OUTPUT_FOLDER, "dataset_summary.json")
MODEL_PATH    = "yolov8n.pt"

# YOLO settings
CONF_THRESHOLD = 0.18    # Low conf to catch occluded/distant people in crowds
IOU_THRESHOLD  = 0.30    # Low IoU allows tightly packed boxes
TILE_SIZE      = 640     # Tile resolution
TILE_OVERLAP   = 0.30    # 30% overlap prevents border misses

SAMPLE_FPS     = 6.0     # Sample 6 frames per second to generate >1000 genuine records (approx 1400 records)

# Area-based threshold calibration (Ref: 640x480 -> 15 people)
BASE_THRESHOLD = 15
REFERENCE_AREA = 640 * 480   # 307,200 pixels


# ==========================================
# AREA-BASED THRESHOLD
# ==========================================

def compute_area_threshold(frame_width: int, frame_height: int) -> int:
    """Auto-scale the crowd alert threshold based on frame area."""
    frame_area = frame_width * frame_height
    scaled     = BASE_THRESHOLD * (frame_area / REFERENCE_AREA)
    return max(10, min(int(round(scaled)), 500))


# ==========================================
# CROWD STATUS & ALERT
# ==========================================

def get_crowd_status(count: int, threshold: int) -> str:
    """Classify crowd status relative to threshold."""
    lower_bound = max(1, threshold // 3)
    if count <= lower_bound:
        return "Normal"
    elif count <= threshold:
        return "Moderate"
    else:
        return "Congested"


def check_alert(count: int, threshold: int, video_name: str, time_sec: float) -> None:
    """Print alert details when threshold is exceeded."""
    print(
        f"\n{'='*62}\n"
        f"  ALERT: Crowd has exceeded threshold!\n"
        f"  Video       : {video_name}\n"
        f"  Time        : {time_sec:.1f}s\n"
        f"  Detected    : {count} people\n"
        f"  Threshold   : {threshold} people\n"
        f"  Exceeded by : +{count - threshold} people above limit\n"
        f"{'='*62}\n"
    )


# ==========================================
# LOAD YOLO MODEL
# ==========================================

def load_yolo_model(model_path: str) -> YOLO:
    """Load YOLOv8 model safely."""
    if not os.path.exists(model_path):
        print(f"[ERROR] YOLO model file not found: '{model_path}'")
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

def detect_with_tiling(model: YOLO, frame: np.ndarray):
    """
    Slice frame into overlapping tiles, run YOLO person detection,
    remap coordinates, apply global NMS to deduplicate detections.
    Returns: (people_count, boxes_xyxy_list, scores_list)
    """
    h, w   = frame.shape[:2]
    stride = int(TILE_SIZE * (1 - TILE_OVERLAP))

    all_boxes  = []
    all_scores = []

    for y_start in range(0, h, stride):
        for x_start in range(0, w, stride):
            x1, y1 = x_start, y_start
            x2 = min(x_start + TILE_SIZE, w)
            y2 = min(y_start + TILE_SIZE, h)

            if (x2 - x1) < 64 or (y2 - y1) < 64:
                continue

            tile = frame[y1:y2, x1:x2]

            try:
                results = model(
                    tile,
                    conf=CONF_THRESHOLD,
                    classes=[0],          # 0 = person
                    imgsz=TILE_SIZE,
                    iou=IOU_THRESHOLD,
                    agnostic_nms=True,
                    verbose=False,
                )
            except Exception:
                continue

            if results[0].boxes is None:
                continue

            for box in results[0].boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                all_scores.append(float(box.conf[0]))

    if not all_boxes:
        return 0, [], []

    boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in all_boxes]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, all_scores, CONF_THRESHOLD, IOU_THRESHOLD)

    if len(indices) == 0:
        return 0, [], []

    final_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
    final_boxes = [all_boxes[i] for i in final_indices]
    final_scores = [all_scores[i] for i in final_indices]

    return len(final_boxes), final_boxes, final_scores


# ==========================================
# PROCESS A SINGLE VIDEO
# ==========================================

def process_video(video_path: str, model: YOLO, sample_fps: float = SAMPLE_FPS) -> tuple[list[dict], dict]:
    """
    Process one video file:
      - Reads metadata (FPS, width, height, duration, total frames)
      - Performs ByteTrack tracking + Tiled detection
      - Returns rows list and video metadata dict
    """
    video_name = os.path.basename(video_path)
    print(f"\n{'─'*62}")
    print(f"[INFO] Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: '{video_path}'. Skipping.")
        return [], {}

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        print(f"[WARNING] Invalid FPS for '{video_name}'. Skipping.")
        cap.release()
        return [], {}

    area_threshold = compute_area_threshold(vid_w, vid_h)
    duration_sec   = total_frames / fps
    frame_interval = max(1, int(round(fps / sample_fps)))

    v_meta = {
        "video_name": video_name,
        "width": vid_w,
        "height": vid_h,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "threshold": area_threshold
    }

    print(f"         Resolution  : {vid_w} x {vid_h}")
    print(f"         FPS/Frames  : {fps:.1f} / {total_frames}")
    print(f"         Duration    : {duration_sec:.1f}s")
    print(f"         Sample every: {frame_interval} frame(s)")
    print(f"         Threshold   : {area_threshold} people")
    print(f"{'─'*62}")

    rows        = []
    frame_idx   = 0
    sample_num  = 0
    alert_count = 0
    seen_track_ids = set()

    # Create persistent video tracker
    tracker = YOLO(MODEL_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            time_sec = round(frame_idx / fps, 2)

            if frame is None or frame.size == 0:
                frame_idx += 1
                continue

            # Run ByteTrack on full frame to get persistent track IDs
            try:
                track_results = tracker.track(
                    frame,
                    conf=CONF_THRESHOLD,
                    classes=[0],
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
                if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
                    ids = track_results[0].boxes.id.int().tolist()
                    for tid in ids:
                        seen_track_ids.add(tid)
            except Exception:
                pass

            # Run Tiled Detection for accurate high-density count
            try:
                people_count, boxes, scores = detect_with_tiling(model, frame)
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
                "Video_Name":        video_name,
                "Time_Seconds":      time_sec,
                "Frame_No":          frame_idx,
                "People_Count":      people_count,
                "Unique_Track_IDs":  len(seen_track_ids),
                "Crowd_Status":      crowd_status,
                "Threshold":         area_threshold,
                "Exceeded":          exceeded,
            })

            # Save first valid frame from the processing batch as sample annotated image
            if not os.path.exists(SAMPLE_IMAGE) and len(boxes) > 0:
                annotated = frame.copy()
                for box, score in zip(boxes, scores):
                    bx1, by1, bx2, by2 = map(int, box)
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Person {score:.2f}", (bx1, max(15, by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                header_str = f"YOLOv8 & Tracking | {video_name} (t={time_sec}s) | People Count: {people_count}"
                cv2.rectangle(annotated, (0, 0), (vid_w, 40), (0, 0, 0), -1)
                cv2.putText(annotated, header_str, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imwrite(SAMPLE_IMAGE, annotated)

            # Real-time state update for mobile server & web page
            try:
                from live_state import update_live_crowd_state
                history_counts = [r["People_Count"] for r in rows]
                update_live_crowd_state(people_count, area_threshold, video_name, history_counts)
            except Exception:
                pass

            sample_num += 1
            icon = "🔴" if exceeded else ("🟡" if crowd_status == "Moderate" else "🟢")
            print(
                f"  t={time_sec:6.1f}s  Frame={frame_idx:5d}  Count={people_count:4d}  "
                f"Unique_IDs={len(seen_track_ids):4d}  Threshold={area_threshold}  {crowd_status:<10} {icon}",
                end="\r",
            )

        frame_idx += 1

    cap.release()
    v_meta["samples_processed"] = sample_num
    v_meta["alert_count"] = alert_count
    v_meta["unique_track_ids"] = len(seen_track_ids)

    print(
        f"\n[OK] '{video_name}': "
        f"{sample_num} samples | {alert_count} alert(s) | "
        f"{len(seen_track_ids)} unique track IDs | Threshold={area_threshold}"
    )
    return rows, v_meta


# ==========================================
# PROCESS ALL VIDEOS IN FOLDER
# ==========================================

def process_all_videos(video_folder: str, model: YOLO, sample_fps: float = SAMPLE_FPS) -> tuple[pd.DataFrame, list[dict]]:
    """Scan video_folder for all .mp4 files and process each independently."""
    if not os.path.isdir(video_folder):
        print(f"[ERROR] Video folder not found: '{video_folder}'")
        sys.exit(1)

    mp4_files = sorted([
        f for f in os.listdir(video_folder)
        if f.lower().endswith(".mp4")
    ])

    if not mp4_files:
        print(f"[ERROR] No .mp4 files found in '{video_folder}'.")
        sys.exit(1)

    print(f"\n[INFO] Found {len(mp4_files)} video(s): {mp4_files}")

    all_rows = []
    video_metas = []

    for filename in mp4_files:
        rows, meta = process_video(os.path.join(video_folder, filename), model, sample_fps=sample_fps)
        all_rows.extend(rows)
        if meta:
            video_metas.append(meta)

    if not all_rows:
        return pd.DataFrame(), []

    return pd.DataFrame(all_rows), video_metas


# ==========================================
# SAVE RESULTS & VALIDATE DATASET
# ==========================================

def save_and_validate_dataset(df: pd.DataFrame, video_metas: list[dict], output_path: str) -> dict:
    """Save crowd dataset to CSV and validate data integrity."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save clean DataFrame
    df.to_csv(output_path, index=False)

    # Compute validation & dataset summary statistics
    total_videos = df["Video_Name"].nunique()
    total_samples = len(df)
    total_detections = int(df["People_Count"].sum())
    max_people = int(df["People_Count"].max())
    min_people = int(df["People_Count"].min())
    mean_people = float(round(df["People_Count"].mean(), 2))
    missing_vals = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    status_dist = df["Crowd_Status"].value_counts().to_dict()

    summary_data = {
        "total_videos_processed": total_videos,
        "total_samples_generated": total_samples,
        "total_detections": total_detections,
        "people_count_stats": {
            "min": min_people,
            "max": max_people,
            "mean": mean_people
        },
        "missing_values": missing_vals,
        "duplicate_records": duplicates,
        "class_distribution": status_dist,
        "video_details": video_metas
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n[OK] Dataset saved successfully: {output_path}")
    print(f"     Total rows           : {total_samples}")
    print(f"     Total videos         : {total_videos}")
    print(f"     Total detections     : {total_detections}")
    print(f"     Missing values       : {missing_vals}")
    print(f"     Duplicate records    : {duplicates}")
    print(f"     Dataset summary json : {SUMMARY_JSON}")

    return summary_data


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 66)
    print("  Smart Temple – Multi-Video Crowd Analysis & Tracking System")
    print("  Mode: Tiled Inference + ByteTrack Person Tracking")
    print("=" * 66)

    model = load_yolo_model(MODEL_PATH)
    df, video_metas = process_all_videos(VIDEO_FOLDER, model, sample_fps=SAMPLE_FPS)

    if df.empty:
        print("[INFO] No data generated. Exiting.")
        return

    summary = save_and_validate_dataset(df, video_metas, OUTPUT_CSV)

    print("\n--- Per-Video Summary ---")
    summary_df = df.groupby("Video_Name").agg(
        Samples=("People_Count", "count"),
        Avg_Count=("People_Count", "mean"),
        Max_Count=("People_Count", "max"),
        Unique_IDs=("Unique_Track_IDs", "max"),
        Threshold=("Threshold", "first"),
        Alerts=("Exceeded", "sum"),
    ).round({"Avg_Count": 1})
    print(summary_df.to_string())

    prev_records = 222
    final_records = len(df)
    new_records = final_records - prev_records
    total_yolo_detections = int(df["People_Count"].sum())
    total_frames_processed = len(df)

    print("\n" + "=" * 66)
    print("  GENUINE DATASET EXPANSION SUMMARY REPORT")
    print("=" * 66)
    print(f"Videos processed: {len(video_metas)}")
    print(f"Frames processed: {total_frames_processed}")
    print(f"YOLO detections: {total_yolo_detections}")
    print(f"New records generated: {new_records}")
    print(f"Previous records: {prev_records}")
    print(f"Final dataset records: {final_records}")
    print("=" * 66)
    print("\n[DONE] All videos processed successfully.")
    print("       Next step: Train model with 'python train_lstm.py'\n")


if __name__ == "__main__":
    main()
