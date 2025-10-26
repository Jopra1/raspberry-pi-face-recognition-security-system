from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import face_recognition
import os
import numpy as np
import shutil
import time
from typing import List

app = FastAPI()

# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Global known faces storage
# ----------------------------
known_encodings = []
known_names = []
root_dataset_path = "dataset"

def load_known_faces():
    """Load all known faces from dataset folder."""
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    if not os.path.exists(root_dataset_path):
        os.makedirs(root_dataset_path)

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

# Initial load of existing faces
load_known_faces()

# ----------------------------
# Webcam setup
# ----------------------------
video_capture = cv2.VideoCapture(0)
process_this_frame = True
i = 0

def gen_frames():
    global process_this_frame, i
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if process_this_frame:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
                name = "Illegal Alien"
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                # Draw rectangles and labels
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        i += 1
        if i == 5:
            i = 0
            process_this_frame = True
        else:
            process_this_frame = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/start-camera")
def start_camera():
    return {"status": "Camera Running"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")



# Upload endpoint for multiple images
@app.post("/upload-images")
async def upload_images(name: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Upload multiple images for a person.
    Saves them in /dataset/<name>/ and updates encodings list immediately.
    """
    os.makedirs(root_dataset_path, exist_ok=True)
    person_folder = os.path.join(root_dataset_path, name)
    os.makedirs(person_folder, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = os.path.join(person_folder, f"{int(time.time())}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

       

    return {"status": "success", "message": f"{len(saved_files)} files uploaded and face encodings updated.", "files": saved_files}
