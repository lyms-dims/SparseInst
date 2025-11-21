import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm

def convert_deepcrack_to_coco(root_dir, set_name):
    """
    Converts DeepCrack dataset to COCO format.
    Args:
        root_dir: Root directory of the dataset (e.g., 'dataset')
        set_name: 'train' or 'test'
    """
    
    # Define paths
    img_dir = os.path.join(root_dir, f"{set_name}_img")
    lab_dir = os.path.join(root_dir, f"{set_name}_lab")
    
    # Output file
    output_dir = os.path.join(root_dir, "annotations")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"instances_{set_name}.json")
    
    # COCO structure
    coco_output = {
        "info": {
            "description": "DeepCrack Dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": "2025-11-21"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "crack", "supercategory": "defect"}
        ]
    }
    
    # Get image files
    img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    img_files.sort()
    
    annotation_id = 1
    
    print(f"Converting {set_name} set...")
    for i, img_path in enumerate(tqdm(img_files)):
        # Image info
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        image_id = i + 1
        
        img_info = {
            "id": image_id,
            "file_name": filename, # Relative path will be handled by dataset mapper or we assume flat structure
            "width": width,
            "height": height,
            "date_captured": "",
            "license": 0,
            "coco_url": "",
            "flickr_url": ""
        }
        coco_output["images"].append(img_info)
        
        # Label info
        # Assuming label has same basename but might be .png
        basename = os.path.splitext(filename)[0]
        # DeepCrack labels are usually .png
        lab_path = os.path.join(lab_dir, basename + ".png")
        
        if not os.path.exists(lab_path):
             # Try .jpg just in case
            lab_path = os.path.join(lab_dir, basename + ".jpg")
            
        if os.path.exists(lab_path):
            mask = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
            # Threshold to binary
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 50: # Filter small noise
                    continue
                
                # Flatten contour coordinates
                segmentation = contour.flatten().tolist()
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1, # Crack
                    "segmentation": [segmentation],
                    "area": cv2.contourArea(contour),
                    "bbox": [x, y, w, h],
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation)
                annotation_id += 1
        else:
            print(f"Warning: Label not found for {filename}")

    with open(output_file, 'w') as f:
        json.dump(coco_output, f)
    
    print(f"Saved {output_file}")

if __name__ == "__main__":
    root = "dataset"
    convert_deepcrack_to_coco(root, "train")
    convert_deepcrack_to_coco(root, "test")
