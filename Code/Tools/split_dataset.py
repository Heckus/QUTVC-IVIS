# split_dataset.py

import os
import random
import shutil
import argparse
import yaml

def split_data(data_dir, train_ratio=0.8):
    """
    Splits the collected data into training and validation sets and updates the YAML file.

    Args:
        data_dir (str): The root directory of the dataset.
        train_ratio (float): The proportion of data to be used for training.
    """
    print(f"[INFO] Starting dataset split in '{data_dir}' with a {train_ratio:.0%} train ratio.")

    # --- 1. Define Paths ---
    source_images_dir = os.path.join(data_dir, "images")
    source_labels_dir = os.path.join(data_dir, "labels")

    if not os.path.isdir(source_images_dir) or not os.path.isdir(source_labels_dir):
        print(f"[ERROR] Source 'images' and 'labels' directories not found in '{data_dir}'.")
        print("[INFO] Run the 'create_volleyball_dataset.py' script first.")
        return

    train_img_dir = os.path.join(data_dir, "train", "images")
    train_lbl_dir = os.path.join(data_dir, "train", "labels")
    val_img_dir = os.path.join(data_dir, "valid", "images")
    val_lbl_dir = os.path.join(data_dir, "valid", "labels")

    # --- 2. Create Destination Directories ---
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    print("[INFO] Created train and valid directories.")

    # --- 3. Get and Shuffle Files ---
    # Get a list of all image filenames without the extension
    all_files = [os.path.splitext(f)[0] for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_files) # IMPORTANT: Shuffle to ensure random distribution

    # --- 4. Calculate Split and Move Files ---
    split_index = int(len(all_files) * train_ratio)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    def move_files(file_list, dest_img_dir, dest_lbl_dir):
        moved_count = 0
        for filename in file_list:
            # Construct source paths
            img_src = os.path.join(source_images_dir, filename + ".jpg") # Assuming .jpg, adjust if needed
            lbl_src = os.path.join(source_labels_dir, filename + ".txt")

            # Move image if it exists
            if os.path.exists(img_src):
                shutil.move(img_src, dest_img_dir)
            
            # Move label if it exists
            if os.path.exists(lbl_src):
                shutil.move(lbl_src, dest_lbl_dir)

            moved_count +=1
        return moved_count

    print(f"[INFO] Moving {len(train_files)} files to the train set...")
    move_files(train_files, train_img_dir, train_lbl_dir)

    print(f"[INFO] Moving {len(val_files)} files to the valid set...")
    move_files(val_files, val_img_dir, val_lbl_dir)
    
    # --- 5. Clean Up Empty Source Directories ---
    # Check if original source directories are empty and remove them
    if not os.listdir(source_images_dir):
        os.rmdir(source_images_dir)
    if not os.listdir(source_labels_dir):
        os.rmdir(source_labels_dir)
    print("[INFO] Cleaned up original source directories.")


    # --- 6. Update data.yaml File ---
    yaml_path = os.path.join(data_dir, "data.yaml")
    abs_data_dir = os.path.abspath(data_dir)

    yaml_data = {
        'path': abs_data_dir,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('valid', 'images'),
        'names': {
            0: 'volleyball'
        }
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f"[SUCCESS] Updated 'data.yaml' to point to new train/valid directories.")
    print("[INFO] Dataset split complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a YOLO dataset into training and validation sets.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the dataset directory (containing 'images' and 'labels' folders).")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training data (e.g., 0.8 for 80%).")
    
    args = parser.parse_args()
    
    if not 0 < args.train_ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1.")
        
    split_data(args.data_dir, args.train_ratio)