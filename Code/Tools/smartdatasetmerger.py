# merge_multiple_datasets.py

"""
Example Usage:
python merge_multiple_datasets.py \
    --datasets ../ball_dataset/data.yaml ../actions_dataset/data.yaml ../another_dataset/data.yaml \
    --output ./final_merged_dataset
"""

import os
import yaml
import shutil
from tqdm import tqdm
import argparse

def get_full_path(yaml_path, relative_path):
    """Constructs an absolute path from a YAML file's location and a relative path within it."""
    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    return os.path.join(base_dir, relative_path)

def process_and_copy_files(split_path, dest_dir, class_map, dataset_prefix):
    """
    Copies images and re-indexes label files for a given data split (train/valid).

    Args:
        split_path (str): The source path for the images in the split.
        dest_dir (str): The destination directory for the split (e.g., './merged_dataset/train').
        class_map (dict): A mapping from old class indices to new ones.
        dataset_prefix (str): A unique prefix to add to filenames to avoid overwrites.
    """
    if not os.path.exists(split_path):
        print(f"Warning: Source path not found, skipping: {split_path}")
        return 0

    image_dir = split_path
    label_dir = split_path.replace('images', 'labels')

    dest_image_dir = os.path.join(dest_dir, 'images')
    dest_label_dir = os.path.join(dest_dir, 'labels')
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    copied_count = 0

    for image_name in tqdm(image_files, desc=f"Processing {os.path.basename(dest_dir)} from {dataset_prefix}"):
        source_image_path = os.path.join(image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        source_label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(source_label_path):
            continue

        new_image_name = f"{dataset_prefix}_{image_name}"
        new_label_name = f"{dataset_prefix}_{label_name}"

        shutil.copy(source_image_path, os.path.join(dest_image_dir, new_image_name))

        new_lines = []
        with open(source_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    old_index = int(parts[0])
                    new_index = class_map.get(old_index)
                    if new_index is not None:
                        new_line = f"{new_index} {' '.join(parts[1:])}"
                        new_lines.append(new_line)

        if new_lines:
            dest_label_path = os.path.join(dest_label_dir, new_label_name)
            with open(dest_label_path, 'w') as f:
                f.write('\n'.join(new_lines))
            copied_count += 1
            
    return copied_count

def get_class_names_from_config(names_field):
    """Safely extracts class name strings from either a list or a dictionary."""
    if isinstance(names_field, dict):
        return list(names_field.values())
    elif isinstance(names_field, list):
        return names_field
    return []

def main(args):
    # --- 1. Load All YAML files ---
    data_configs = []
    all_original_names = []
    for yaml_path in args.datasets:
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                data_configs.append(config)
                all_original_names.append(get_class_names_from_config(config['names']))
        except Exception as e:
            print(f"Error reading or parsing YAML file {yaml_path}: {e}")
            return

    # --- 2. Create the master class list ---
    master_class_set = set()
    for names_list in all_original_names:
        master_class_set.update(names_list)
    master_class_list = sorted(list(master_class_set))
    
    print("--- Master Class List ---")
    for i, name in enumerate(master_class_list):
        print(f"{i}: {name}")
    print("-------------------------")

    # --- 3. Create class mappings for each dataset ---
    class_mappings = []
    for names_list in all_original_names:
        class_map = {old_idx: master_class_list.index(name) for old_idx, name in enumerate(names_list)}
        class_mappings.append(class_map)

    # --- 4. Create Output Directories ---
    output_dir = args.output
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)

    # --- 5. Process and copy files for all datasets ---
    for i, yaml_path in enumerate(args.datasets):
        data_config = data_configs[i]
        class_map = class_mappings[i]
        
        print(f"\n--- Processing Dataset {i+1}: {os.path.basename(yaml_path)} ---")
        prefix = os.path.splitext(os.path.basename(yaml_path))[0]
        
        train_path = get_full_path(yaml_path, data_config['train'])
        valid_path = get_full_path(yaml_path, data_config.get('val', data_config['train'])) # Use train if val not specified
        
        process_and_copy_files(train_path, os.path.join(output_dir, 'train'), class_map, prefix)
        process_and_copy_files(valid_path, os.path.join(output_dir, 'valid'), class_map, prefix)

    # --- 6. Create the final merged YAML file ---
    final_yaml_path = os.path.join(output_dir, 'data.yaml')
    names_dict = {i: name for i, name in enumerate(master_class_list)}
    
    final_config = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': len(master_class_list),
        'names': names_dict
    }

    with open(final_yaml_path, 'w') as f:
        yaml.dump(final_config, f, sort_keys=False, default_flow_style=False)
        
    print("\n--- Merging Complete! ---")
    print(f"Merged dataset created at: {output_dir}")
    print(f"New master YAML file at: {final_yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge multiple YOLOv8 datasets with potentially shared classes.")
    parser.add_argument('--datasets', nargs='+', required=True, help='A list of paths to the dataset YAML files (e.g., data1.yaml data2.yaml ...).')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory for the merged dataset.')
    
    args = parser.parse_args()
    main(args)