from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
import os
import time
import json
import argparse
import numpy as np
from collections import deque

# Parse command line parameters
parser = argparse.ArgumentParser(description="Smart Temple Live Crowd Detection")
parser.add_argument("--threshold", type=int, default=20, help="Custom alert threshold for this camera view")
parser.add_argument("--source", type=str, default=None, help="Camera index (0, 1) or path to video file (e.g. videos/v1.mp4)")
args, _ = parser.parse_known_args()

CLI_THRESHOLD = args.threshold
CLI_SOURCE = args.source

print(f"Loading YOLO model... (Initial Threshold: {CLI_THRESHOLD})")
model = YOLO("yolov8n.pt")


# ==========================================
# OPEN CAMERA OR VIDEO FALLBACK
# ==========================================

IS_VIDEO_FILE = False
VIDEO_FILE_PATH = None

def find_working_camera():
    global IS_VIDEO_FILE, VIDEO_FILE_PATH
    
    # User specified explicit video or camera source
    if CLI_SOURCE is not None:
        if CLI_SOURCE.isdigit():
            idx = int(CLI_SOURCE)
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap
        else:
            if os.path.exists(CLI_SOURCE):
                print(f"[INFO] Opening specified video source: {CLI_SOURCE}")
                IS_VIDEO_FILE = True
                VIDEO_FILE_PATH = CLI_SOURCE
                return cv2.VideoCapture(CLI_SOURCE)

    # Fast check for physical webcams
    for index in [0, 1]:
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[OK] Connected to webcam index {index}!")
                    return cap
                cap.release()
        except Exception:
            continue

    # Fallback to sample video in videos/ directory if no physical camera
    sample_videos = ["videos/v1.mp4", "videos/v2.mp4"]
    for vid in sample_videos:
        if os.path.exists(vid):
            print(f"\n[INFO] No physical webcam detected on this device.")
            print(f"[INFO] Automatically streaming sample video '{vid}' for live demonstration!")
            IS_VIDEO_FILE = True
            VIDEO_FILE_PATH = vid
            return cv2.VideoCapture(vid)

    return cv2.VideoCapture(0)

cap = find_working_camera()

if not cap.isOpened():
    print("ERROR: Cannot open camera or sample video.")
    exit()

if not IS_VIDEO_FILE:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Detection stream started successfully!")


# ==========================================
# OUTPUT FOLDER
# ==========================================

os.makedirs("output", exist_ok=True)

csv_file = "output/crowd_data.csv"

if os.path.exists(csv_file):
    os.remove(csv_file)

if os.path.exists("output/stop_camera.signal"):
    os.remove("output/stop_camera.signal")


# ==========================================
# LARGE WINDOW - NO FULLSCREEN ZOOM
# ==========================================

window_name = "Smart Temple Crowd Management System"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Large window
cv2.resizeWindow(window_name, 1200, 800)


# ==========================================
# VARIABLES
# ==========================================

last_saved_time = ""

display_count = 0
last_detection_time = time.time()

PERSON_HOLD_TIME = 1
count_window = deque(maxlen=15)


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        if IS_VIDEO_FILE:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
        else:
            print("Unable to read camera frame. Retrying...")
            time.sleep(0.1)
            continue

    # Keep original camera view
    frame = cv2.resize(frame, (640, 480))


    # ==========================================
    # YOLO PERSON DETECTION
    # ==========================================

    results = model(
        frame,
        conf=0.50,
        classes=[0],
        imgsz=416,
        verbose=False
    )


    # ==========================================
    # COUNT PEOPLE
    # ==========================================

    current_count = 0

    if results[0].boxes is not None:
        current_count = len(results[0].boxes)

    count_window.append(current_count)


    # ==========================================
    # STABLE COUNT (TEMPORAL MEDIAN FILTERING)
    # ==========================================

    current_timestamp = time.time()
    smoothed_count = int(round(float(np.median(count_window))))

    if smoothed_count > 0:
        display_count = smoothed_count
        last_detection_time = current_timestamp

    elif current_timestamp - last_detection_time > PERSON_HOLD_TIME:
        display_count = 0


    # ==========================================
    # DYNAMIC CROWD STATUS & THRESHOLD
    # ==========================================

    CROWD_THRESHOLD = CLI_THRESHOLD
    config_file = "output/live_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
                CROWD_THRESHOLD = int(cfg.get("threshold", CLI_THRESHOLD))
        except Exception:
            pass

    normal_limit = max(1, CROWD_THRESHOLD // 3)

    if display_count <= normal_limit:
        crowd_status = "Normal"
        status_message = f"Status: Normal (Max {CROWD_THRESHOLD})"

    elif display_count <= CROWD_THRESHOLD:
        crowd_status = "Moderate"
        status_message = f"WARNING: Crowd Increasing (Max {CROWD_THRESHOLD})"

    else:
        crowd_status = "Congested"
        status_message = f"ALERT: CROWD EXCEEDED THRESHOLD ({display_count} > {CROWD_THRESHOLD})!"


    # ==========================================
    # DRAW YOLO BOXES
    # ==========================================

    annotated_frame = results[0].plot()


    # ==========================================
    # DISPLAY INFORMATION
    # ==========================================

    cv2.putText(
        annotated_frame,
        f"People Count: {display_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Crowd Status: {crowd_status}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        status_message,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    # ==========================================
    # SAVE DATA TO CSV ONCE PER SECOND
    # ==========================================

    current_time = datetime.now().strftime("%H:%M:%S")

    if current_time != last_saved_time:

        new_data = pd.DataFrame([{
            "Timestamp": current_time,
            "People_Count": display_count,
            "Crowd_Status": crowd_status
        }])

        file_exists = os.path.exists(csv_file)

        new_data.to_csv(
            csv_file,
            mode="a",
            header=not file_exists,
            index=False
        )

        last_saved_time = current_time

        print(
            f"Saved: {current_time} | "
            f"Count: {display_count} | "
            f"Status: {crowd_status}"
        )


    # ==========================================
    # SHOW VIDEO & SAVE FRAME FOR DASHBOARD
    # ==========================================

    cv2.imwrite("output/live_frame.jpg", annotated_frame)
    cv2.imshow(window_name, annotated_frame)


    # Press Q to stop OR stop signal from Web Dashboard
    if (cv2.waitKey(1) & 0xFF == ord("q")) or os.path.exists("output/stop_camera.signal"):
        print("Stopping detection...")
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

# Remove live frame image so dashboard clears frozen picture
if os.path.exists("output/live_frame.jpg"):
    try:
        os.remove("output/live_frame.jpg")
    except Exception:
        pass

print("Detection stopped.")
print(f"Data saved in: {csv_file}")