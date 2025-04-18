import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from tensorflow.keras.models import load_model

# Load ASL model
model = load_model('asl_model.h5')

# ASL class labels
class_labels = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
])

# Image size
img_size = 224

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Prediction buffer (for smoothing)
prediction_buffer = deque(maxlen=15)  # Last 15 predictions

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    display_label = ""
    display_conf = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.size == 0:
                continue

            hand_crop = cv2.resize(hand_crop, (img_size, img_size))
            hand_crop = hand_crop.astype("float32") / 255.0
            hand_crop = np.expand_dims(hand_crop, axis=0)

            prediction = model.predict(hand_crop, verbose=0)
            pred_index = np.argmax(prediction)
            pred_label = class_labels[pred_index]
            confidence = np.max(prediction) * 100

            # Update prediction buffer
            prediction_buffer.append(pred_label)

            # Most common prediction in buffer
            most_common_pred = Counter(prediction_buffer).most_common(1)[0]
            display_label = most_common_pred[0]
            display_conf = confidence

            # Draw rectangle and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{display_label} ({display_conf:.1f}%)',
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

    cv2.imshow("ASL Detection (Smoothing ON)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
