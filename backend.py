from fastapi import FastAPI, JSONResponse
from fastapi.responses import StreamingResponse
import uvicorn
import face_recognition
import cv2
import os
import numpy as np
import pickle
from gpiozero import AngularServo
from time import sleep, time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Load known encodings ======
encodings_file = "encodings.pkl"
known_encodings = []
known_names = []

if os.path.exists(encodings_file):
    with open(encodings_file, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]

# ====== Servo setup ======
servo = AngularServo(18, min_angle=0, max_angle=180,
                     min_pulse_width=0.0005, max_pulse_width=0.0024)
servo.angle = 0
sleep(1)

# ====== Camera ======
video_capture = None
servo_open = False
last_seen_time = 0

# ====== Store last detected faces ======
last_face_locations = []
last_face_names = []

@app.get("/start-camera")
def start_camera():
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return JSONResponse(status_code=500, content={"message": "Failed to open camera"})
    return {"message": "Camera started successfully"}


def generate_frames():
    global servo_open, last_seen_time, last_face_locations, last_face_names

    frame_count = 0
    detection_interval = 5  # run detection every 5 frames

    while True:
        if video_capture is None:
            break

        ret, frame = video_capture.read()
        if not ret:
            break

        # Run detection only every `detection_interval` frames
        if frame_count % detection_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        last_seen_time = time()
                        if not servo_open:
                            servo.angle = 90
                            servo_open = True
                            print("Opening Door")
                face_names.append(name)

            # Update last detected faces
            last_face_locations = face_locations
            last_face_names = face_names

        # Draw last known boxes on every frame
        for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Close servo if no face seen for 5 seconds
        if servo_open and time() - last_seen_time > 5:
            servo.angle = 0
            servo_open = False
            print("Closing Door")

        frame_count += 1

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video_feed")
def video_feed():
    if video_capture is None:
        return JSONResponse(status_code=400, content={"message": "Camera not started"})
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
