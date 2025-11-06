from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import face_recognition
import cv2
import os
import numpy as np
import pickle
import shutil
from gpiozero import AngularServo
from time import sleep, time
from typing import List

app = FastAPI()

# ====== CORS Middleware ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Dataset & Encodings ======
encodings_file = "encodings.pkl"
root_dataset_path = "dataset"
known_encodings = []
known_names = []

# Load existing encodings if present
if os.path.exists(encodings_file):
    print("Loading existing encodings...")
    with open(encodings_file, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
else:
    # Create new encodings if file doesn't exist
    if not os.path.exists(root_dataset_path):
        os.makedirs(root_dataset_path)
    for person_name in os.listdir(root_dataset_path):
        person_folder = os.path.join(root_dataset_path, person_name)
        if os.path.isdir(person_folder):
            for file in os.listdir(person_folder):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(person_folder, file)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)
    with open(encodings_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print("Encodings saved to encodings.pkl")

# ====== Servo Setup ======
servo = AngularServo(18, min_angle=0, max_angle=180,
                     min_pulse_width=0.0005, max_pulse_width=0.0024)
servo.angle = 0
sleep(1)

# ====== Camera & State ======
video_capture = None
servo_open = False
last_seen_time = 0
last_face_locations = []
last_face_names = []

# ====== Start Camera ======
@app.get("/start-camera")
def start_camera():
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return JSONResponse(status_code=500, content={"message": "Failed to open camera"})
    return {"message": "Camera started successfully"}

# ====== Frame Generator ======
def generate_frames():
    global servo_open, last_seen_time, last_face_locations, last_face_names
    frame_count = 0
    detection_interval = 5

    while True:
        if video_capture is None:
            break
        ret, frame = video_capture.read()
        if not ret:
            break

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
                            print("✅ Door Opening")
                face_names.append(name)

            last_face_locations = face_locations
            last_face_names = face_names

        for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if servo_open and time() - last_seen_time > 5:
            servo.angle = 0
            servo_open = False
            print("🔒 Door Closing")

        frame_count += 1
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ====== Video Feed Endpoint ======
@app.get("/video_feed")
def video_feed():
    if video_capture is None:
        return JSONResponse(status_code=400, content={"message": "Camera not started"})
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# ====== Upload Images Endpoint ======
@app.post("/upload-images")
async def upload_images(name: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Upload multiple images for a person and save them under /dataset/<name>/.
    """
    os.makedirs(root_dataset_path, exist_ok=True)
    person_folder = os.path.join(root_dataset_path, name)
    os.makedirs(person_folder, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = os.path.join(person_folder, f"{int(time())}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    # Optionally, update known_encodings immediately (or later by reloading encodings)
    # load_known_faces()  # uncomment if you want instant recognition

    return {"status": "success",
            "message": f"{len(saved_files)} files uploaded for {name}.",
            "files": saved_files}

# ====== Run Server ======
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
