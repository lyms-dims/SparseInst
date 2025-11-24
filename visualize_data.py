import random
import cv2
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from tools.register_deepcracks_dataset import register_deepcracks_dataset

# ============================================================================
# 1. REGISTER DATASET (Crucial Step)
# ============================================================================
# We must register the dataset again in this script so Detectron2 "sees" it.
# Ensure these paths match your folder structure exactly.
register_deepcracks_dataset(
    dataset_name="deepcracks_train",
    json_file="datasets/DeepCrack/annotations/train.json",
    image_root="datasets/DeepCrack/train_img"
)

# You can register val/test here too if you want to visualize them
# register_deepcracks_dataset("deepcracks_val", "datasets/DeepCrack/annotations/val.json", "datasets/DeepCrack/val_img")

# ============================================================================
# 2. VISUALIZATION LOGIC
# ============================================================================

def visualize_samples(dataset_name, num_samples=5, output_dir="vis_output"):
    """
    Pick random samples from the dataset and save visualizations.
    """
    # Check if registered
    if dataset_name not in DatasetCatalog.list():
        print(f"Error: {dataset_name} is not registered.")
        return

    # Get dataset metadata
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {num_samples} visualizations to '{output_dir}'...")

    # Pick random samples
    # Use min() to handle cases where dataset has fewer images than num_samples
    samples_to_take = min(len(dataset_dicts), num_samples)
    
    for d in random.sample(dataset_dicts, samples_to_take):
        img_path = d["file_name"]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error: Could not read {img_path}")
            continue

        # Initialize Visualizer
        # scale=1.0 means original size. Increase/decrease if image is too small/large
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        
        # Draw the ground truth annotations
        vis = visualizer.draw_dataset_dict(d)
        
        # Save result
        file_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"vis_{file_name}")
        cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    visualize_samples("deepcracks_train", num_samples=5)