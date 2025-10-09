import face_recognition
import cv2
import os
import numpy as np # Note: np import is not strictly needed for this specific change but is often useful

# Step 1: Configuration
known_encodings = []
known_names = []
# The main directory containing subfolders for each person
root_dataset_path = "dataset"

print(f"[INFO] Looking for people folders in: {root_dataset_path}")

# Step 2: Load known faces from subfolders
# Iterate through all items (folders) in the root_dataset_path
for person_name in os.listdir(root_dataset_path):
    person_folder_path = os.path.join(root_dataset_path, person_name)
    
    # Check if the item is a directory (a person's folder)
    if os.path.isdir(person_folder_path):
        # person_name is now the folder name (e.g., 'joel', 'shiv')
        print(f"[INFO] Loading images for: {person_name}")

        # Iterate through all image files inside the person's folder
        for file in os.listdir(person_folder_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(person_folder_path, file)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    # Append the encoding and use the folder name as the person's name
                    known_encodings.append(encodings[0])
                    known_names.append(person_name) 
                    # print(f"    - Loaded encoding from: {file}") # Optional debug print
                else:
                    print(f"[WARNING] No face found in {file} (in {person_name}'s folder), skipping.")

print("[INFO] Loaded encodings for:", known_names)

# Step 3: Initialize webcam (rest of the code is largely unchanged)
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("[ERROR] Failed to open webcam.")
    exit(1)

i = 0    
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    if process_this_frame:
        
        # Step 4: Detect faces in frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize for speed if necessary, but keep the original logic for now
        face_locations = face_recognition.face_locations(rgb_frame, model="hog") 
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Step 5: Compare with known faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
            name = "Illegal Alien"

            # Pick the best match
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances) # np.argmin is more explicit
                if matches[best_match_index]:
                    # Use the name from the folder
                    name = known_names[best_match_index]

            # Step 6: Draw box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show video
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        process_this_frame = False

    i += 1

    # Only process every 5th frame for performance
    if i == 5:
        i = 0
        process_this_frame = True

video_capture.release()
cv2.destroyAllWindows()