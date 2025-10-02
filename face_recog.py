import face_recognition
import cv2
import os
import numpy as np

# Get absolute path for "images" folder (next to script)
images_path = os.path.join(os.path.dirname(__file__), "images")
print(f"[DEBUG] Looking for images folder at: {images_path}")

# Load known faces
known_face_encodings = []
known_face_names = []

# Check if images_path exists
if not os.path.exists(images_path):
    print(f"[ERROR] Images folder not found at: {images_path}")
    exit(1)

for file in os.listdir(images_path):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(images_path, file)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:  # only add if a face was detected
            known_face_encodings.append(encodings[0])
            # File name (without extension) = person's name
            name = os.path.splitext(file)[0]
            known_face_names.append(name)
            print(f"[INFO] Loaded encoding for: {name}")
        else:
            print(f"[WARNING] No face found in {file}, skipping.")

if not known_face_encodings:
    print("[WARNING] No known faces loaded. Please add images with detectable faces to the 'images' folder.")
else:
    print(" Loaded encodings for:", known_face_names)

# Start webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("[ERROR] Failed to open webcam.")
    exit(1)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    # Resize frame for speed (1/4 size)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
 
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Pick the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale face locations back to full size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Show video feed
    cv2.imshow("Face Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()