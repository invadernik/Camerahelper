import cv2
import time
import os
import numpy as np


# Path to the model files
prototxt_path = os.path.sep.join(['models', 'MobileNetSSD_deploy.prototxt'])
model_path = os.path.sep.join(['models', 'MobileNetSSD_deploy.caffemodel'])

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Specify the video file path
video_path = '/workspaces/codespaces-blank/Project Space 01/Camera Tracker/Running - 294.mp4'  # Replace with your video file

# Initialize video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Could not open video.")
    exit()

# Get frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

# List to store timestamps where people are detected
timestamps = []

print("[INFO] Starting video processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Calculate timestamp in seconds
    timestamp = frame_count / fps

    # Prepare the frame for object detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    person_detected = False

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            person_detected = True

            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * \
                np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box around the person
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if person_detected:
        # Convert timestamp to HH:MM:SS format
        formatted_time = time.strftime(
            '%H:%M:%S', time.gmtime(timestamp))
        timestamps.append(formatted_time)
        print(f"[INFO] Person detected at {formatted_time}")

    # Optional: Display the frame (can be commented out if not needed)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Remove duplicate timestamps
unique_timestamps = sorted(set(timestamps), key=timestamps.index)

# Save timestamps to a text file
with open('detection_timestamps.txt', 'w') as f:
    for ts in unique_timestamps:
        f.write(f"{ts}\n")

print("[INFO] Video processing completed.")
print(f"[INFO] Detected person timestamps saved to 'detection_timestamps.txt'")
