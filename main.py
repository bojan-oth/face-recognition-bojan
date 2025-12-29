import cv2
import os
import numpy as np
from collections import deque

dataset_path = r"dataset"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []
label_dict = {}
current_label = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_dict[current_label] = person
    label = current_label
    current_label += 1

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in detected:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            faces.append(face)
            labels.append(label)

faces = np.array(faces, dtype="uint8")
labels = np.array(labels, dtype="int32")

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)
print("Modelot e treniran.")

cap = cv2.VideoCapture(0)

face_buffers = {}
THRESHOLD = 60
BUFFER_SIZE = 7

def find_closest_buffer(center, buffers, max_distance=50):
    for key, data in buffers.items():
        cx, cy = data['center']
        if abs(cx - center[0]) < max_distance and abs(cy - center[1]) < max_distance:
            return key
    return None

next_face_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.1, 5)

    current_ids = []

    for (x, y, w, h) in detected:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, distance = model.predict(face)

        center = (x + w//2, y + h//2)
        face_id = find_closest_buffer(center, face_buffers)

        if face_id is None:
            face_id = next_face_id
            face_buffers[face_id] = {'buffer': deque(maxlen=BUFFER_SIZE), 'center': center}
            next_face_id += 1
        else:
            face_buffers[face_id]['center'] = center

        face_buffers[face_id]['buffer'].append(distance)
        avg_distance = sum(face_buffers[face_id]['buffer']) / len(face_buffers[face_id]['buffer'])
        confidence = max(0, int(100 - avg_distance))

        if avg_distance < THRESHOLD:
            name = label_dict[label]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} {confidence}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
