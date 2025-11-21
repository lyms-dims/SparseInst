"""
Register DeepCracks dataset with Detectron2 for SparseInst training.

This script registers the converted COCO format dataset with Detectron2's
MetadataCatalog and DatasetCatalog.
"""
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def register_deepcracks_dataset(
    dataset_name,
    json_file,
    image_root,
    thing_classes=None
):
    """
    Register DeepCracks dataset with Detectron2.
    
    Args:
        dataset_name: Name to register the dataset (e.g., "deepcracks_train")
        json_file: Path to COCO format JSON annotation file
        image_root: Path to directory containing images
        thing_classes: List of class names (default: ["crack"])
    """
    if thing_classes is None:
        thing_classes = ["crack"]
    
    # Register the dataset
    register_coco_instances(
        dataset_name,
        {},
        json_file,
        image_root
    )
    
    # Set metadata
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = thing_classes
    metadata.evaluator_type = "coco"  # Use COCO evaluator
    
    print(f"Registered dataset: {dataset_name}")
    print(f"  JSON file: {json_file}")
    print(f"  Image root: {image_root}")
    print(f"  Classes: {thing_classes}")
    
    return metadata


def main():
    """
    Example usage - modify these paths according to your dataset structure.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Register DeepCracks dataset with Detectron2")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name to register (e.g., deepcracks_train)"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        required=True,
        help="Path to COCO format JSON annotation file"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Path to directory containing images"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isfile(args.json_file):
        raise ValueError(f"JSON file does not exist: {args.json_file}")
    if not os.path.isdir(args.image_root):
        raise ValueError(f"Image root directory does not exist: {args.image_root}")
    
    # Register dataset
    register_deepcracks_dataset(
        args.dataset_name,
        args.json_file,
        args.image_root
    )
    
    print("\nDataset registration complete!")
    print(f"You can now use '{args.dataset_name}' in your config file.")


if __name__ == "__main__":
    main()

