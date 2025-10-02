# liveness_check.py
import cv2
import face_recognition
import numpy as np
import time
from collections import deque
import math

# ---- Config ----
EAR_THRESHOLD = 0.23        # eye aspect ratio threshold (tune per camera)
EAR_CONSEC_FRAMES = 2      # consecutive frames EAR must be below threshold to count as a blink
LIVENESS_WINDOW = 4.0      # seconds within which required actions must happen
HEAD_YAW_DEG_MIN = 10.0    # minimum yaw degrees to count as a head turn
RECOGNITION_TOLERANCE = 0.45

# Load known face encodings / names as in your existing script
# Example:
# known_face_encodings = [...]
# known_face_names = [...]
# For demo, I'll leave them as empty lists:
known_face_encodings = []
known_face_names = []

# Helper functions
def eye_aspect_ratio(eye):
    # eye: list of (x,y) points (6 points for a dlib-style eye)
    # compute vertical and horizontal distances
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear

def get_head_pose(image_points, size):
    # 3D model points of facial features (generic)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4,1))  # assume no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    # Convert rotation vector to Euler angles (in degrees)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = [float(x) for x in euler_angles]
    return pitch, yaw, roll

# Open camera
cap = cv2.VideoCapture(0)
blink_counter = 0
blinked = False
start_time = None
events = deque()  # store ('blink' or ('head',deg), timestamp)

print("Starting liveness check. Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb = small[:, :, ::-1]

    # Face detection + recognition (using your existing flow)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            continue  # ignore unknown faces for unlock

        # Get landmarks (face_recognition returns landmarks in original image scale)
        landmarks = face_recognition.face_landmarks(rgb, [(top,right,bottom,left)])[0]

        # EAR (blink) using left_eye/right_eye lists
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0

        # Blink logic
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                # registered a blink
                ts = time.time()
                events.append(('blink', ts))
                print(f"[{name}] blink at {ts}")
                # if start_time not set, set it
                if start_time is None:
                    start_time = ts
                blinked = True
            blink_counter = 0

        # Head pose detection: build the 2D image points from landmarks
        # take nose tip, chin, left/right eye corners, mouth corners
        # face_recognition landmarks give many mouth/eye points - pick sensible ones:
        nose_point = landmarks['nose_tip'][len(landmarks['nose_tip'])//2]
        chin_point = landmarks['chin'][len(landmarks['chin'])//2]
        left_eye_corner = landmarks['left_eye'][0]
        right_eye_corner = landmarks['right_eye'][-1]
        left_mouth = landmarks['top_lip'][0]
        right_mouth = landmarks['top_lip'][-1]

        image_points = np.array([
            nose_point,
            chin_point,
            left_eye_corner,
            right_eye_corner,
            left_mouth,
            right_mouth
        ], dtype=np.float64)

        pitch_yaw_roll = get_head_pose(image_points, small.shape)
        if pitch_yaw_roll is not None:
            pitch, yaw, roll = pitch_yaw_roll
            # If yaw exceeds threshold, record as head movement
            if abs(yaw) > HEAD_YAW_DEG_MIN:
                ts = time.time()
                events.append(('head', yaw, ts))
                print(f"[{name}] head yaw {yaw:.1f} deg at {ts}")
                if start_time is None:
                    start_time = ts

        # Evaluate liveness window
        if start_time is not None:
            now = time.time()
            # discard old events outside window
            while events and (now - events[0][-1]) > LIVENESS_WINDOW:
                events.popleft()
            # Check if both blink and head movement present in events
            has_blink = any(e[0]=='blink' for e in events)
            has_head = any(e[0]=='head' and abs(e[1])>HEAD_YAW_DEG_MIN for e in events)
            if has_blink and has_head:
                print(f"*** Liveness confirmed for {name} — unlocking now! ***")
                # TODO: trigger unlock mechanism here
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

    # show visual feedback (optional)
    cv2.imshow("Liveness", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
