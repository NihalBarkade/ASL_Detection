import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained ASL model
model = load_model('asl_model.h5')

# Define class labels in the same order used during training
class_labels = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
])

# Image size expected by the model
img_size = 224

# Start webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the live prediction.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI) for ASL gesture (You can adjust the box size/location)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI for prediction
    img = cv2.resize(roi, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    pred_index = np.argmax(prediction)
    pred_label = class_labels[pred_index]
    confidence = np.max(prediction) * 100

    # Draw rectangle and show prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'{pred_label} ({confidence:.1f}%)', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("ASL Real-Time Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy
cap.release()
cv2.destroyAllWindows()
