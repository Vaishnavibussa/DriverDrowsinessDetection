import os
import shutil
import random

# Paths
SOURCE_OPEN = "data/MRL/train/Open_Eyes"
SOURCE_CLOSED = "data/MRL/train/Closed_Eyes"
TARGET_DIR = "data_cleaned"

# Create new clean structure
for split in ["train", "val", "test"]:
    for label in ["awake", "drowsy"]:
        os.makedirs(os.path.join(TARGET_DIR, split, label), exist_ok=True)

# Helper function to split and copy images
def split_and_copy(source_folder, label):
    images = os.listdir(source_folder)
    random.shuffle(images)

    total = len(images)
    train_split = int(0.7 * total)
    val_split = int(0.15 * total)

    for idx, img_name in enumerate(images):
        src_path = os.path.join(source_folder, img_name)

        if idx < train_split:
            dest_folder = os.path.join(TARGET_DIR, "train", label)
        elif idx < train_split + val_split:
            dest_folder = os.path.join(TARGET_DIR, "val", label)
        else:
            dest_folder = os.path.join(TARGET_DIR, "test", label)

        shutil.copy(src_path, dest_folder)

# Split Open Eyes -> awake
split_and_copy(SOURCE_OPEN, "awake")

# Split Closed Eyes -> drowsy
split_and_copy(SOURCE_CLOSED, "drowsy")

print("✅ Dataset preparation completed!")
