import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

DATA_FILE = "isl_dataset.csv"

if not os.path.exists(DATA_FILE):
    cols = []
    for i in range(21):
        cols.append(f"x{i}")
    for i in range(21):
        cols.append(f"y{i}")
    cols.append("label")
    pd.DataFrame(columns=cols).to_csv(DATA_FILE, index=False)

label = input("Enter sign label (example: A, B, Hello): ")

cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened() and count < 50:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
            for lm in hand_landmarks.landmark:
                row.append(lm.y)

            row.append(label)
            pd.DataFrame([row]).to_csv(DATA_FILE, mode='a', header=False, index=False)
            count += 1
            print(f"Collected {count}/50")

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
