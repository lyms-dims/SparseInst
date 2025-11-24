import json
import os

def clean_json(json_path, image_dir):
    """
    Removes images from JSON that do not exist on disk.
    """
    print(f"Scanning {json_path}...")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    original_count = len(data['images'])
    valid_images = []
    valid_image_ids = set()
    missing_count = 0

    # 1. Check every image in the JSON
    for img in data['images']:
        file_path = os.path.join(image_dir, img['file_name'])
        
        # Check if file actually exists
        if os.path.exists(file_path):
            valid_images.append(img)
            valid_image_ids.add(img['id'])
        else:
            print(f"  [Missing] Removed entry for: {img['file_name']}")
            missing_count += 1

    # 2. Keep only annotations for valid images
    valid_annotations = [ann for ann in data['annotations'] if ann['image_id'] in valid_image_ids]

    # 3. Update and Save
    if missing_count > 0:
        data['images'] = valid_images
        data['annotations'] = valid_annotations
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Fixed! Removed {missing_count} missing images.")
        print(f"New image count: {len(valid_images)}")
    else:
        print("No missing files found. JSON is already clean.")
    print("-" * 40)

if __name__ == "__main__":
    # Clean Train Set
    clean_json(
        "datasets/DeepCrack/annotations/train.json", 
        "datasets/DeepCrack/train_img"
    )
    
    # Clean Validation Set
    clean_json(
        "datasets/DeepCrack/annotations/val.json", 
        "datasets/DeepCrack/val_img"
    )