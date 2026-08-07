from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, conf= 0.5, verbose=False)

    # Count only persons
    person_count = 0

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            if model.names[cls] == "person":
                person_count += 1

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display count
    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Temple Crowd Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()