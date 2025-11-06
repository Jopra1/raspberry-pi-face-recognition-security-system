from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import face_recognition
import os
import numpy as np
import pickle
from gpiozero import AngularServo, Device
from gpiozero.pins.pigpio import PiGPIOFactory
from threading import Thread
from time import sleep, time

# ----------------------------
# Setup FastAPI
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Servo setup (PiGPIOFactory for stable PWM)
# ----------------------------
Device.pin_factory = PiGPIOFactory()
servo = AngularServo(18, min_angle=0, max_angle=180)
servo.angle = 0

# ----------------------------
# Load known face encodings
# ----------------------------
encodings_file = "encodings.pkl"
known_encodings = []
known_names = []

if os.path.exists(encodings_file):
    with open(encodings_file, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
else:
    # Create encodings from dataset if not exists
    root_dataset_path = "dataset"
    for person_name in os.listdir(root_dataset_path):
        person_folder_path = os.path.join(root_dataset_path, person_name)
        if os.path.isdir(person_folder_path):
            for file in os.listdir(person_folder_path):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(person_folder_path, file)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)
    with open(encodings_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

# ----------------------------
# Camera & global state
# ----------------------------
video_capture = None
last_face_locations = []
last_face_names = []
servo_open = False
last_seen_time = 0
detection_frame_interval = 5  # detect every 5 frames
frame_count = 0
process_frame = True
recognized = False

# ----------------------------
# Background detection thread
# ----------------------------
def detection_loop():
    global last_face_locations, last_face_names, servo_open, last_seen_time, recognized, process_frame, frame_count
    while True:
        if video_capture is None:
            sleep(0.1)
            continue

        ret, frame = video_capture.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % detection_frame_interval != 0:
            continue

        # Resize for faster detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations_small = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings_small = face_recognition.face_encodings(rgb_small, face_locations_small)

        temp_names = []
        face_found = False
        for (top, right, bottom, left), face_encoding in zip(face_locations_small, face_encodings_small):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    face_found = True
                    last_seen_time = time()
            temp_names.append(name)

        # Update global locations (scale up to original frame)
        last_face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations_small]
        last_face_names = temp_names
        recognized = face_found

        # Servo control (only on state change)
        if recognized and not servo_open:
            servo.angle = 90
            servo_open = True
            print("Opening Door")
        elif not recognized and servo_open and time() - last_seen_time > 5:
            servo.angle = 0
            servo_open = False
            print("Closing Door")

        sleep(0.01)  # small sleep to prevent CPU overload

# Start detection thread
Thread(target=detection_loop, daemon=True).start()

# ----------------------------
# FastAPI endpoints
# ----------------------------
@app.get("/start-camera")
def start_camera():
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return {"status": "error", "message": "Failed to open camera"}
    return {"status": "success", "message": "Camera started successfully"}

def generate_frames():
    global last_face_locations, last_face_names, video_capture
    while True:
        if video_capture is None:
            sleep(0.1)
            continue

        ret, frame = video_capture.read()
        if not ret:
            continue

        # Draw face boxes
        for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
