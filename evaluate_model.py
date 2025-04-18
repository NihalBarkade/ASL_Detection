import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from data_loader import get_data_generators

# ---- Load Model ----
model = load_model("asl_model.h5")

# ---- Load Data ----
train_path = 'Dataset/train_split'
val_path = 'Dataset/val_split'
_, val_gen = get_data_generators(train_path, val_path)
class_names = list(val_gen.class_indices.keys())

# ---- Evaluate ----
y_true, y_pred = [], []

for batch_images, batch_labels in val_gen:
    preds = model.predict(batch_images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(batch_labels, axis=1))
    
    if len(y_true) >= val_gen.samples:
        break

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()