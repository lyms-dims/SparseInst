import os
import json
import shutil
import random
from pathlib import Path

def split_dataset(dataset_root, train_json_path, val_ratio=0.2):
    """
    Splits train dataset into train and val.
    Moves images and creates new JSON files.
    """
    print(f"Loading {train_json_path}...")
    with open(train_json_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    # Group annotations by image_id for easy lookup
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Shuffle and split
    random.seed(42)
    random.shuffle(images)
    
    num_val = int(len(images) * val_ratio)
    val_images = images[:num_val]
    train_images = images[num_val:]
    
    print(f"Splitting: {len(train_images)} Train, {len(val_images)} Validation")

    # Helper to build new json
    def build_json(imgs, image_root_src, image_root_dst=None):
        new_anns = []
        for img in imgs:
            # Add associated annotations
            if img['id'] in img_to_anns:
                new_anns.extend(img_to_anns[img['id']])
            
            # Move image if destination provided
            if image_root_dst:
                src_path = os.path.join(image_root_src, img['file_name'])
                dst_path = os.path.join(image_root_dst, img['file_name'])
                
                # Check if file exists before moving
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                elif os.path.exists(dst_path):
                    print(f"Note: {dst_path} already exists.")
                else:
                    print(f"Warning: Image not found {src_path}")
        
        return {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "categories": categories,
            "images": imgs,
            "annotations": new_anns
        }

    # Paths
    root = Path(dataset_root)
    train_img_dir = root / "train_img"
    val_img_dir = root / "val_img"
    val_img_dir.mkdir(exist_ok=True)
    
    # Create Validation JSON and Move Images
    print("Moving validation images...")
    val_data = build_json(val_images, str(train_img_dir), str(val_img_dir))
    val_json_path = root / "annotations" / "val.json"
    with open(val_json_path, 'w') as f:
        json.dump(val_data, f, indent=2)
        
    # Create New Train JSON (images stay in train_img)
    print("Updating train annotations...")
    train_data = build_json(train_images, str(train_img_dir), None)
    
    # Overwrite the old train.json
    with open(train_json_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    print("Done!")
    print(f"Validation images moved to: {val_img_dir}")
    print(f"Created: {val_json_path}")

if __name__ == "__main__":
    # Based on your file structure
    split_dataset(
        dataset_root="datasets/DeepCrack",
        train_json_path="datasets/DeepCrack/annotations/train.json"
    )