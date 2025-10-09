import os
import cv2
import face_recognition
import numpy as np
import time
from collections import deque

# ---- Config ----
EAR_THRESHOLD = 0.23        # eye aspect ratio threshold
EAR_CONSEC_FRAMES = 2       # consecutive frames EAR must be below threshold
LIVENESS_WINDOW = 4.0       # seconds within which required actions must happen
HEAD_YAW_DEG_MIN = 10.0     # minimum yaw degrees to count as head turn
RECOGNITION_TOLERANCE = 0.45

# ---- Helper functions ----
def eye_aspect_ratio(eye):
    """Compute the eye aspect ratio (EAR)"""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0

def get_head_pose(image_points, size):
    """Estimate head pose (pitch, yaw, roll) in degrees"""
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = [float(x) for x in euler_angles]
    return pitch, yaw, roll

# ---- Load known faces ----
images_path = os.path.join(os.path.dirname(__file__), "images")
known_face_encodings = []
known_face_names = []

if not os.path.exists(images_path):
    print(f"[ERROR] Images folder not found at: {images_path}")
    exit(1)

for file in os.listdir(images_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = face_recognition.load_image_file(os.path.join(images_path, file))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(file)[0])
            print(f"[INFO] Loaded encoding for: {os.path.splitext(file)[0]}")
        else:
            print(f"[WARNING] No face found in {file}, skipping.")

if not known_face_encodings:
    print("[ERROR] No known faces loaded. Add images to 'images' folder.")
    exit(1)

# ---- Start webcam ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Failed to open webcam.")
    exit(1)

blink_counter = 0
start_time = None
events = deque()  # store ('blink' or ('head', yaw), timestamp)

print("Starting liveness check. Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb_small = small[:, :, ::-1]

    # ---- FIXED SECTION ----
    face_locations = face_recognition.face_locations(rgb_small)

    face_encodings = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = rgb_small[top:bottom, left:right]
        enc = face_recognition.face_encodings(face_image)
        if enc:
            face_encodings.append(enc[0])
    # ------------------------

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Match with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE)
        name = "Unknown"
        if True in matches:
            best_idx = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            if matches[best_idx]:
                name = known_face_names[best_idx]
        else:
            continue  # ignore unknown faces for unlock

        # Get facial landmarks
        landmarks = face_recognition.face_landmarks(rgb_small, [(top, right, bottom, left)])[0]

        # EAR for blink
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                ts = time.time()
                events.append(('blink', ts))
                print(f"[{name}] blink at {ts}")
                if start_time is None:
                    start_time = ts
            blink_counter = 0

        # Head pose detection
        nose_point = landmarks['nose_tip'][len(landmarks['nose_tip'])//2]
        chin_point = landmarks['chin'][len(landmarks['chin'])//2]
        left_eye_corner = landmarks['left_eye'][0]
        right_eye_corner = landmarks['right_eye'][-1]
        left_mouth = landmarks['top_lip'][0]
        right_mouth = landmarks['top_lip'][-1]

        image_points = np.array([nose_point, chin_point, left_eye_corner, right_eye_corner, left_mouth, right_mouth], dtype=np.float64)
        pose = get_head_pose(image_points, small.shape)
        if pose is not None:
            pitch, yaw, roll = pose
            if abs(yaw) > HEAD_YAW_DEG_MIN:
                ts = time.time()
                events.append(('head', yaw, ts))
                print(f"[{name}] head yaw {yaw:.1f} deg at {ts}")
                if start_time is None:
                    start_time = ts

        # Liveness check
        if start_time is not None:
            now = time.time()
            # Remove old events
            while events and (now - events[0][-1]) > LIVENESS_WINDOW:
                events.popleft()
            has_blink = any(e[0]=='blink' for e in events)
            has_head = any(e[0]=='head' and abs(e[1])>HEAD_YAW_DEG_MIN for e in events)
            if has_blink and has_head:
                print(f"*** Liveness confirmed for {name} — unlocking now! ***")
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

    # Show webcam feed
    cv2.imshow("Liveness", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
