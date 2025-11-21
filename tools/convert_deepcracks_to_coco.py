"""
Convert DeepCracks dataset format to COCO format for SparseInst training.

DeepCracks format:
    - images/ folder containing images
    - masks/ folder containing binary mask images (same filenames)

COCO format:
    - JSON annotation file with images, annotations, and categories
    - Images folder (can be same as input)
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from pycocotools import mask as mask_util


def binary_mask_to_rle(binary_mask):
    """Convert binary mask to RLE format."""
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def find_instances_in_mask(mask):
    """
    Find individual crack instances in a binary mask using connected components.
    Each connected component is treated as a separate instance.
    """
    # Convert to binary if needed
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Threshold to ensure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    instances = []
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get mask for this instance
        instance_mask = (labels == i).astype(np.uint8) * 255
        
        # Get bounding box from stats
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        
        # Skip very small instances (noise)
        if area < 10:  # Minimum area threshold
            continue
        
        # Convert mask to RLE
        rle = binary_mask_to_rle(instance_mask)
        
        instances.append({
            'mask': instance_mask,
            'bbox': [x, y, w, h],
            'area': area,
            'rle': rle
        })
    
    return instances


def convert_deepcracks_to_coco(images_dir, masks_dir, output_json, dataset_name="deepcracks"):
    """
    Convert DeepCracks dataset to COCO format.
    
    Args:
        images_dir: Path to directory containing images
        masks_dir: Path to directory containing mask images
        output_json: Path to output COCO JSON file
        dataset_name: Name of the dataset
    """
    # COCO format structure
    coco_data = {
        "info": {
            "description": f"{dataset_name} dataset for crack detection",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "categories": [
            {
                "id": 1,
                "name": "crack",
                "supercategory": "defect"
            }
        ],
        "images": [],
        "annotations": []
    }
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images")
    
    image_id = 1
    annotation_id = 1
    
    for img_path in tqdm(image_files, desc="Converting images"):
        # Find corresponding mask
        mask_path = Path(masks_dir) / img_path.name
        
        # Try different extensions if exact match not found
        if not mask_path.exists():
            found = False
            for ext in image_extensions:
                alt_mask_path = Path(masks_dir) / (img_path.stem + ext)
                if alt_mask_path.exists():
                    mask_path = alt_mask_path
                    found = True
                    break
            if not found:
                print(f"Warning: No mask found for {img_path.name}, skipping...")
                continue
        
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping...")
            continue
        
        height, width = img.shape[:2]
        
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}, skipping...")
            continue
        
        # Resize mask if dimensions don't match
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        }
        coco_data["images"].append(image_info)
        
        # Find instances in mask
        instances = find_instances_in_mask(mask)
        
        # Add annotations for each instance
        for instance in instances:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # crack category
                "segmentation": instance['rle'],
                "area": instance['area'],
                "bbox": instance['bbox'],  # [x, y, width, height]
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    # Save COCO JSON
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Output saved to: {output_json}")
    
    return coco_data


def main():
    parser = argparse.ArgumentParser(description="Convert DeepCracks dataset to COCO format")
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=True,
        help="Path to directory containing mask images"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to output COCO JSON file"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="deepcracks",
        help="Name of the dataset (default: deepcracks)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.images_dir):
        raise ValueError(f"Images directory does not exist: {args.images_dir}")
    if not os.path.isdir(args.masks_dir):
        raise ValueError(f"Masks directory does not exist: {args.masks_dir}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    
    # Convert
    convert_deepcracks_to_coco(
        args.images_dir,
        args.masks_dir,
        args.output_json,
        args.dataset_name
    )


if __name__ == "__main__":
    main()

