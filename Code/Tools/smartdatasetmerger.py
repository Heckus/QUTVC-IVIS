import os
import yaml
import shutil
import argparse
import random
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
IMG_FORMATS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo'}

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_image_files(path):
    """Recursively find all images in a directory."""
    path = Path(path)
    files = []
    if path.is_file():
        if path.suffix.lower() in IMG_FORMATS:
            files.append(path)
    elif path.is_dir():
        for p in path.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_FORMATS:
                files.append(p)
    return files

def process_dataset_files(image_files, dest_img_dir, dest_lbl_dir, class_map, prefix):
    """
    Copies images and creates re-mapped label files.
    """
    for src_img_path in image_files:
        # Define paths
        src_img_path = Path(src_img_path)
        
        # Determine label path (YOLO standard: replace /images/ with /labels/ and ext with .txt)
        # We try a few common locations for the label file
        possible_label_dirs = [
            src_img_path.parent.parent / 'labels',  # standard yolo (.../dataset/images/file.jpg -> .../dataset/labels/file.txt)
            src_img_path.parent / 'labels',         # adjacent folder
            src_img_path.parent                     # same folder
        ]
        
        src_lbl_path = None
        for p in possible_label_dirs:
            candidate = p / (src_img_path.stem + '.txt')
            if candidate.exists():
                src_lbl_path = candidate
                break
        
        # New Names
        new_filename = f"{prefix}_{src_img_path.name}"
        dest_img_path = dest_img_dir / new_filename
        dest_lbl_path = dest_lbl_dir / (Path(new_filename).stem + ".txt")

        # 1. Copy Image
        shutil.copy2(src_img_path, dest_img_path)

        # 2. Process Label (if exists)
        new_lines = []
        if src_lbl_path and src_lbl_path.exists():
            with open(src_lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: # class x y w h
                        old_idx = int(parts[0])
                        # Remap index
                        if old_idx in class_map:
                            new_idx = class_map[old_idx]
                            new_lines.append(f"{new_idx} {' '.join(parts[1:])}")
        
        # 3. Write Label File (Always write, even if empty, to confirm it's a checked image)
        # Note: Some trainers prefer no file for background, some prefer empty file. 
        # Writing an empty file is generally safer for tracking.
        if new_lines:
            with open(dest_lbl_path, 'w') as f:
                f.write('\n'.join(new_lines))
        # If no lines, we don't strictly need to write the file for YOLOv8 (it infers bg), 
        # but creating an empty file explicitly defines it as a negative sample.
        elif not src_lbl_path:
             with open(dest_lbl_path, 'w') as f:
                pass # Empty file for background image

def main(args):
    print(f"--- Starting Merge of {len(args.datasets)} Datasets ---")
    
    # 1. Gather all configs and build Master Class List
    configs = []
    all_class_names = set()
    
    # Read all YAMLs first to unify classes
    for yaml_path in args.datasets:
        cfg = load_yaml(yaml_path)
        configs.append({'path': Path(yaml_path), 'cfg': cfg})
        
        # Handle dictionary or list format for names
        names = cfg.get('names', {})
        if isinstance(names, dict):
            names = names.values()
        
        # Normalize to lowercase to avoid 'Ball' vs 'ball' conflicts
        for n in names:
            all_class_names.add(str(n).lower())

    # Create Master List (sorted for consistency)
    master_classes = sorted(list(all_class_names))
    print(f"\nMaster Class List ({len(master_classes)} classes): {master_classes}")

    # 2. Prepare Output Directories
    out_dir = Path(args.output)
    if out_dir.exists():
        print(f"Warning: Output directory {out_dir} already exists.")
        # Optional: shutil.rmtree(out_dir)
    
    dirs = {
        'train': {'images': out_dir / 'train' / 'images', 'labels': out_dir / 'train' / 'labels'},
        'val':   {'images': out_dir / 'valid' / 'images', 'labels': out_dir / 'valid' / 'labels'}
    }
    
    for split in dirs:
        for dtype in dirs[split]:
            dirs[split][dtype].mkdir(parents=True, exist_ok=True)

    # 3. Collect ALL data (Images + Mappings)
    # We gather everything into a list first, then shuffle and split globally.
    all_data_entries = [] # Stores tuples: (image_path, class_map, prefix)

    for entry in configs:
        yaml_path = entry['path']
        cfg = entry['cfg']
        prefix = yaml_path.stem # e.g. "video1_data"
        
        # Build local->master mapping
        local_names = cfg.get('names', {})
        local_map = {} # old_id -> new_id
        
        if isinstance(local_names, list):
            for idx, name in enumerate(local_names):
                if str(name).lower() in master_classes:
                    local_map[idx] = master_classes.index(str(name).lower())
        elif isinstance(local_names, dict):
            for idx, name in local_names.items():
                if str(name).lower() in master_classes:
                    local_map[int(idx)] = master_classes.index(str(name).lower())

        # Find image folders
        # We check keys 'train', 'val', and 'test' in the yaml to find where images are
        keys_to_check = ['train', 'val', 'test']
        
        found_images = []
        yaml_parent = yaml_path.parent
        
        for key in keys_to_check:
            if key in cfg:
                path_val = cfg[key]
                # path_val can be a string or list
                if isinstance(path_val, str):
                    path_val = [path_val]
                
                for p in path_val:
                    # Resolve path relative to YAML
                    full_p = (yaml_parent / p).resolve()
                    found_images.extend(get_image_files(full_p))

        # Add to global list
        for img_path in found_images:
            all_data_entries.append({
                'image': img_path,
                'map': local_map,
                'prefix': prefix
            })

    # 4. Shuffle and Split
    total_images = len(all_data_entries)
    print(f"\nCollected {total_images} total images. Shuffling and splitting ({args.split_ratio} train)...")
    
    random.seed(42) # Deterministic split
    random.shuffle(all_data_entries)
    
    split_idx = int(total_images * args.split_ratio)
    train_set = all_data_entries[:split_idx]
    val_set = all_data_entries[split_idx:]
    
    # 5. Process Files
    def process_batch(dataset, split_name):
        dest_img = dirs[split_name]['images']
        dest_lbl = dirs[split_name]['labels']
        
        print(f"Processing {split_name} set ({len(dataset)} images)...")
        for item in tqdm(dataset):
            process_dataset_files(
                [item['image']], 
                dest_img, 
                dest_lbl, 
                item['map'], 
                item['prefix']
            )

    process_batch(train_set, 'train')
    process_batch(val_set, 'val')

    # 6. Create Final YAML
    final_yaml_path = out_dir / 'data.yaml'
    final_names_map = {i: name for i, name in enumerate(master_classes)}
    
    final_config = {
        'path': str(out_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': len(master_classes),
        'names': final_names_map
    }

    with open(final_yaml_path, 'w') as f:
        yaml.dump(final_config, f, sort_keys=False)

    print(f"\n--- Merge Complete ---")
    print(f"Output: {out_dir}")
    print(f"Config: {final_yaml_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge multiple datasets into one YOLOv8 dataset with auto-splitting.")
    parser.add_argument('--datasets', nargs='+', required=True, help='List of paths to data.yaml files')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Ratio of images to use for training (default 0.8)')
    
    args = parser.parse_args()
    main(args)