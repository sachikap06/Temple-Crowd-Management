from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
import os
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Output folder
os.makedirs("output", exist_ok=True)

# CSV file
csv_file = "output/crowd_data.csv"

# Create empty CSV at the beginning of a new run
if os.path.exists(csv_file):
    os.remove(csv_file)

last_saved_time = ""

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera.")
        break

    # YOLO detection
    results = model(frame, conf=0.5, verbose=False)

    # Count persons
    person_count = 0

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            if model.names[cls] == "person":
                person_count += 1

    # Crowd status
    if person_count <= 5:
        crowd_status = "Normal"

    elif person_count <= 15:
        crowd_status = "Moderate"

    else:
        crowd_status = "Congested"

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display people count
    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display crowd status
    cv2.putText(
        annotated_frame,
        f"Crowd Status: {crowd_status}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Current time
    current_time = datetime.now().strftime("%H:%M:%S")

    # Save once every second
    if current_time != last_saved_time:

        new_data = pd.DataFrame([{
            "Timestamp": current_time,
            "People_Count": person_count,
            "Crowd_Status": crowd_status
        }])

        # Add header only for the first record
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
            f"Count: {person_count} | "
            f"Status: {crowd_status}"
        )

    # Show video
    cv2.imshow("Temple Crowd Detection", annotated_frame)

    # Q stops the camera
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera
cap.release()
cv2.destroyAllWindows()

print("Detection stopped.")