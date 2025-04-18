import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('asl_model.h5')

# Directory containing one test image per class
test_image_dir = r'D:\InternShip_Projects\ASL_DETECTION_2\Dataset\asl_alphabet_test'

# Class labels (ensure same order as during model training)
class_labels = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
])

# Model input image size
img_size = 224

# Accuracy tracking
correct = 0
total = len(class_labels)

for label in class_labels:
    img_path = os.path.join(test_image_dir, f'{label}_test.jpg')
    
    if not os.path.exists(img_path):
        print(f"Missing: {img_path}")
        continue

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # model expects batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    pred_label = class_labels[pred_index]

    print(f"True: {label} | Predicted: {pred_label} | {'✅' if pred_label == label else '❌'}")

    if pred_label == label:
        correct += 1

# Final accuracy
print(f"\nAccuracy on 29-class test set: {correct}/{total} = {correct / total * 100:.2f}%")

