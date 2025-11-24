import argparse
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from glob import glob

# Setup detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Add root to path so we can find sparseinst and tools
sys.path.append(".") 
from sparseinst import add_sparse_inst_config

# ============================================================================
# Register Datasets (Crucial for class names)
# ============================================================================
try:
    from tools.register_deepcracks_dataset import register_deepcracks_dataset
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from register_deepcracks_dataset import register_deepcracks_dataset

# We register the validation set just to get the metadata (class names like "crack")
try:
    register_deepcracks_dataset(
        dataset_name="deepcracks_val",
        json_file="datasets/DeepCrack/annotations/val.json",
        image_root="datasets/DeepCrack/val_img"
    )
except AssertionError:
    pass 
# ============================================================================

def setup_cfg(config_file, weights, confidence, size):
    """Setup configuration for a model."""
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(config_file)
    
    # Use the final trained weights
    cfg.MODEL.WEIGHTS = weights
    
    # Use GPU if available
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set confidence threshold (0.0 to 1.0)
    # Only show cracks if model is > 40% confident
    cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = confidence
    
    # Standard image sizing for inference
    cfg.INPUT.MIN_SIZE_TEST = size
    
    cfg.freeze()
    return cfg

def process_image(predictor, image_path, output_path, metadata):
    """Process a single image and save the result."""
    # Read Image
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Could not read image: {image_path}")
        return False

    # Run Detection
    outputs = predictor(im)
    
    # Visualize
    instances = outputs["instances"].to("cpu")
    num_instances = len(instances)
    
    # Draw on image
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(instances)
    
    # Save Result
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    print(f"  ✓ {os.path.basename(image_path)} -> {os.path.basename(output_path)} ({num_instances} cracks)")
    return True

def get_image_files(input_path):
    """Get all image files from a directory or return single file in a list."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        return [str(input_path)]
    elif input_path.is_dir():
        # Directory - find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(str(input_path / ext)))
            image_files.extend(glob(str(input_path / ext.upper())))
        return sorted(image_files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

def get_model_config_path(model_path):
    """Get config file path for a model. Checks model directory first, then looks for common config files."""
    model_dir = Path(model_path).parent
    config_in_dir = model_dir / "config.yaml"
    
    if config_in_dir.exists():
        return str(config_in_dir)
    
    # Try to infer from model path
    if "gabor" in str(model_path).lower():
        # Look for gabor config
        gabor_config = Path("configs/sparse_inst_r50_giam_crack_gabor.yaml")
        if gabor_config.exists():
            return str(gabor_config)
    
    # Default to regular crack config
    default_config = Path("configs/sparse_inst_r50_giam_crack.yaml")
    if default_config.exists():
        return str(default_config)
    
    raise ValueError(f"Could not find config file for model: {model_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Detect cracks on image(s) with one or more models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python tools/predict.py --config-file configs/sparse_inst_r50_giam_crack.yaml \\
      --weights output/crack_model_turbo/model_final.pth --input datasets/DeepCrack/test_img

  # Compare multiple models
  python tools/predict.py --models output/crack_model_turbo/model_final.pth \\
      output/crack_gabor_turbo/model_final.pth --input datasets/DeepCrack/test_img \\
      --output-dir comparison_results
        """
    )
    
    # Model selection: either single model (old way) or multiple models (new way)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--weights", help="Path to model_final.pth (single model mode)")
    model_group.add_argument("--models", nargs="+", help="Paths to multiple model_final.pth files (comparison mode)")
    
    parser.add_argument("--config-file", default=None, help="Path to config.yaml (optional, auto-detected if not provided)")
    parser.add_argument("--config-files", nargs="+", help="Config files for each model (must match --models count)")
    parser.add_argument("--input", required=True, help="Path to input image file or folder")
    parser.add_argument("--output", default=None, help="Output path (file for single image, dir for batch)")
    parser.add_argument("--output-dir", default=None, help="Output directory for batch processing (overrides --output for folders)")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--size", type=int, default=512, help="Resize shortest edge to this (default 512)")
    parser.add_argument("--model-names", nargs="+", help="Custom names for models (for output folders, must match --models count)")
    
    args = parser.parse_args()

    # Determine model configuration
    if args.models:
        # Multi-model mode
        model_paths = [Path(p) for p in args.models]
        model_names = args.model_names if args.model_names else [p.parent.name for p in model_paths]
        
        if len(model_names) != len(model_paths):
            print(f"Error: Number of model names ({len(model_names)}) must match number of models ({len(model_paths)})")
            return
        
        # Get config files
        if args.config_files:
            if len(args.config_files) != len(model_paths):
                print(f"Error: Number of config files ({len(args.config_files)}) must match number of models ({len(model_paths)})")
                return
            config_files = args.config_files
        else:
            # Auto-detect config files
            config_files = [get_model_config_path(str(p)) for p in model_paths]
        
        print("=" * 70)
        print("MULTI-MODEL COMPARISON MODE")
        print("=" * 70)
        print(f"Models to compare: {len(model_paths)}")
        for i, (model_path, model_name, config_file) in enumerate(zip(model_paths, model_names, config_files), 1):
            print(f"  {i}. {model_name}")
            print(f"     Model: {model_path}")
            print(f"     Config: {config_file}")
        print("=" * 70)
        
        # Load all models
        predictors = []
        for model_path, model_name, config_file in zip(model_paths, model_names, config_files):
            print(f"\nLoading model: {model_name}...")
            cfg = setup_cfg(config_file, str(model_path), args.confidence, args.size)
            predictor = DefaultPredictor(cfg)
            predictors.append((predictor, model_name))
            print(f"  ✓ {model_name} loaded successfully!")
        
    else:
        # Single model mode (backward compatibility)
        model_path = Path(args.weights)
        model_name = model_path.parent.name
        
        # Get config file
        if args.config_file:
            config_file = args.config_file
        else:
            config_file = get_model_config_path(str(model_path))
        
        print(f"Loading model from {args.weights}...")
        cfg = setup_cfg(config_file, args.weights, args.confidence, args.size)
        predictor = DefaultPredictor(cfg)
        predictors = [(predictor, model_name)]
        print("Model loaded successfully!")

    # Get image files to process
    image_files = get_image_files(args.input)
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {args.input}")
        return
    
    print(f"\nFound {len(image_files)} image(s) to process")
    print("-" * 50)

    # Get metadata for "crack" label
    metadata = MetadataCatalog.get("deepcracks_val")
    
    input_path = Path(args.input)
    is_single_file = input_path.is_file()
    
    if len(predictors) == 1:
        # Single model mode
        predictor, model_name = predictors[0]
        
        if is_single_file:
            # Single file mode
            output_path = args.output if args.output else "prediction_result.jpg"
            print(f"Processing: {args.input}")
            process_image(predictor, args.input, output_path, metadata)
            print("-" * 50)
            print(f"Done! Result saved to: {output_path}")
        else:
            # Batch mode
            if args.output_dir:
                output_dir = Path(args.output_dir)
            elif args.output:
                output_dir = Path(args.output)
            else:
                output_dir = Path(args.input).parent / "predictions"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_dir}")
            print("-" * 50)
            
            # Process each image
            successful = 0
            failed = 0
            
            for i, image_path in enumerate(image_files, 1):
                print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
                
                # Generate output filename
                image_name = Path(image_path).stem
                output_path = output_dir / f"{image_name}_pred.jpg"
                
                if process_image(predictor, image_path, str(output_path), metadata):
                    successful += 1
                else:
                    failed += 1
            
            print("-" * 50)
            print(f"Batch processing complete!")
            print(f"  Successful: {successful}/{len(image_files)}")
            if failed > 0:
                print(f"  Failed: {failed}/{len(image_files)}")
            print(f"Results saved to: {output_dir}")
    
    else:
        # Multi-model comparison mode
        if args.output_dir:
            output_base_dir = Path(args.output_dir)
        elif args.output:
            output_base_dir = Path(args.output)
        else:
            output_base_dir = Path(args.input).parent / "model_comparison"
        
        output_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput base directory: {output_base_dir}")
        print("-" * 50)
        
        # Create subdirectories for each model and initialize counters
        model_output_dirs = {}
        total_successful = {}
        total_failed = {}
        
        for predictor, model_name in predictors:
            model_dir = output_base_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_output_dirs[model_name] = model_dir
            total_successful[model_name] = 0
            total_failed[model_name] = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            image_name = Path(image_path).stem
            
            for predictor, model_name in predictors:
                print(f"  Running {model_name}...")
                output_dir = model_output_dirs[model_name]
                output_path = output_dir / f"{image_name}_pred.jpg"
                
                if process_image(predictor, image_path, str(output_path), metadata):
                    total_successful[model_name] += 1
                else:
                    total_failed[model_name] += 1
        
        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE!")
        print("=" * 70)
        for predictor, model_name in predictors:
            successful = total_successful[model_name]
            failed = total_failed[model_name]
            print(f"\n{model_name}:")
            print(f"  Successful: {successful}/{len(image_files)}")
            if failed > 0:
                print(f"  Failed: {failed}/{len(image_files)}")
            print(f"  Results saved to: {model_output_dirs[model_name]}")
        print("=" * 70)

if __name__ == "__main__":
    main()