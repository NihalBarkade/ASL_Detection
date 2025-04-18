import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, val_split=0.1):
    classes = os.listdir(source_dir)
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)

        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_split))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create target folders
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

# Paths
SOURCE = 'Dataset/asl_alphabet_train'
TRAIN = 'Dataset/train_split'
VAL = 'Dataset/val_split'

split_data(SOURCE, TRAIN, VAL)
