# src/model_trainer.py
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_data_generators

from trainer_visualizer import plot_training_history
from evaluate_model import evaluate_model

# Paths
train_path = 'Dataset/train_split'
val_path = 'Dataset/val_split'
model_save_path = 'asl_model.h5'

# Load data
train_gen, val_gen = get_data_generators(train_path, val_path)

# Get number of classes
num_classes = train_gen.num_classes

# Base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

# plot training history
plot_training_history(history)

# Evaluate model
class_names = list(train_gen.class_indices.keys())
evaluate_model(model, val_gen, class_names)
