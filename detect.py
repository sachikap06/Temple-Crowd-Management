from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
import os
import time

# ==========================================
# SMART TEMPLE CROWD MANAGEMENT SYSTEM
# ==========================================

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")


# ==========================================
# OPEN CAMERA
# ==========================================

print("Opening webcam...")

def find_working_camera():
    for index in [0, 1]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Successfully connected to camera index {index}!")
                return cap
            cap.release()
    print("Defaulting to camera index 0.")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap

cap = find_working_camera()

if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened successfully!")


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


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret or frame is None or frame.mean() < 1:
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
        conf=0.35,
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


    # ==========================================
    # STABLE COUNT
    # ==========================================

    current_timestamp = time.time()

    if current_count > 0:
        display_count = current_count
        last_detection_time = current_timestamp

    elif current_timestamp - last_detection_time > PERSON_HOLD_TIME:
        display_count = 0


    # ==========================================
    # CROWD STATUS
    # ==========================================

    if display_count <= 5:
        crowd_status = "Normal"
        status_message = "Status: Normal"

    elif display_count <= 15:
        crowd_status = "Moderate"
        status_message = "WARNING: Crowd Increasing"

    else:
        crowd_status = "Congested"
        status_message = "ALERT: HIGH CROWD!"


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