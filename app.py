import gradio as gr
import cv2
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Setup detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Add root to path
sys.path.append(".")
from sparseinst import add_sparse_inst_config

# Register dataset for metadata
try:
    from tools.register_deepcracks_dataset import register_deepcracks_dataset
    try:
        register_deepcracks_dataset(
            dataset_name="deepcracks_val",
            json_file="datasets/DeepCrack/annotations/val.json",
            image_root="datasets/DeepCrack/val_img"
        )
    except AssertionError:
        pass
except ImportError:
    print("Warning: Could not import register_deepcracks_dataset")

# Global predictor cache
predictors_cache = {}

def setup_cfg(config_file, weights, confidence, size, force_cpu=False):
    """Setup configuration for a model."""
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    
    # Allow forcing CPU mode (useful if GPU hangs)
    if force_cpu:
        cfg.MODEL.DEVICE = "cpu"
    else:
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = confidence
    cfg.INPUT.MIN_SIZE_TEST = size
    cfg.freeze()
    return cfg

def get_available_models():
    """Scan output directory for available trained models."""
    models = {}
    output_dir = Path("output")
    
    if not output_dir.exists():
        return models
    
    # Look for model_final.pth files in output subdirectories
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            model_file = model_dir / "model_final.pth"
            if model_file.exists():
                # Try to find corresponding config
                config_file = model_dir / "config.yaml"
                if not config_file.exists():
                    # Try to infer config
                    if "gabor" in model_dir.name.lower():
                        config_file = Path("configs/sparse_inst_r50_giam_crack_gabor.yaml")
                    else:
                        config_file = Path("configs/sparse_inst_r50_giam_crack.yaml")
                
                if config_file.exists():
                    models[model_dir.name] = {
                        "weights": str(model_file),
                        "config": str(config_file)
                    }
    
    return models

def predict_single_model(image, model_name, model_info, confidence_threshold, image_size, force_cpu):
    """Run prediction with a single model."""
    # Check if files exist
    if not os.path.exists(model_info['weights']):
        raise FileNotFoundError(f"Model weights file not found: {model_info['weights']}")
    if not os.path.exists(model_info['config']):
        raise FileNotFoundError(f"Config file not found: {model_info['config']}")
    
    # Check model file size
    model_size_mb = os.path.getsize(model_info['weights']) / (1024 * 1024)
    print(f"  Model file size: {model_size_mb:.1f} MB")
    
    # Use cache key with device info
    device_suffix = "_cpu" if force_cpu else "_gpu"
    cache_key = model_name + device_suffix
    
    # Load or get cached predictor
    if cache_key not in predictors_cache:
        print(f"  Loading model for the first time...")
        print(f"  This may take 30-60 seconds...")
        cfg = setup_cfg(
            model_info["config"],
            model_info["weights"],
            confidence_threshold,
            image_size,
            force_cpu=force_cpu
        )
        print(f"  Config created, device: {cfg.MODEL.DEVICE}")
        print(f"  Creating predictor...")
        sys.stdout.flush()
        
        predictors_cache[cache_key] = DefaultPredictor(cfg)
        
        print(f"  ✓ Predictor loaded on {cfg.MODEL.DEVICE}!")
    else:
        print(f"  Using cached predictor")
    
    predictor = predictors_cache[cache_key]
    
    # Update confidence threshold if needed
    predictor.cfg.defrost()
    predictor.cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = confidence_threshold
    predictor.cfg.INPUT.MIN_SIZE_TEST = image_size
    predictor.cfg.freeze()
    
    # Convert image to BGR for OpenCV
    if len(image.shape) == 2:
        im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    print(f"  Running detection...")
    outputs = predictor(im)
    
    # Get predictions
    instances = outputs["instances"].to("cpu")
    num_instances = len(instances)
    print(f"  ✓ Found {num_instances} crack instances")
    
    # Visualize
    metadata = MetadataCatalog.get("deepcracks_val")
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(instances)
    
    # Convert back to RGB
    result_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    
    return result_image, num_instances

def predict_crack_comparison(image, confidence_threshold, image_size, force_cpu):
    """Predict cracks using both models for comparison."""
    try:
        print(f"\n{'='*60}")
        print(f"Starting DUAL MODEL comparison...")
        print(f"Confidence: {confidence_threshold}")
        print(f"Image size: {image_size}")
        print(f"Force CPU: {force_cpu}")
        
        if image is None:
            print("Error: No image provided")
            return None, None, "Please upload an image first."
        
        print(f"Image shape: {image.shape}")
        
        # Get available models
        available_models = get_available_models()
        
        if not available_models:
            print("Error: No models found")
            return None, None, "❌ No trained models found in 'output/' directory. Please train models first."
        
        # Find normal and gabor models
        normal_model = None
        gabor_model = None
        
        for model_name, model_info in available_models.items():
            if "gabor" in model_name.lower():
                gabor_model = (model_name, model_info)
            else:
                normal_model = (model_name, model_info)
        
        if not normal_model and not gabor_model:
            return None, None, "❌ No models found. Please ensure you have trained models."
        
        device = "GPU (CUDA)" if torch.cuda.is_available() and not force_cpu else "CPU"
        results = []
        
        # Process normal model
        if normal_model:
            model_name, model_info = normal_model
            print(f"\n[1/2] Processing with NORMAL model: {model_name}")
            print(f"  Weights: {model_info['weights']}")
            print(f"  Config: {model_info['config']}")
            
            result_img, num_cracks = predict_single_model(
                image, model_name, model_info, 
                confidence_threshold, image_size, force_cpu
            )
            results.append(("Normal Model", model_name, num_cracks, result_img))
        
        # Process gabor model
        if gabor_model:
            model_name, model_info = gabor_model
            print(f"\n[2/2] Processing with GABOR model: {model_name}")
            print(f"  Weights: {model_info['weights']}")
            print(f"  Config: {model_info['config']}")
            
            result_img, num_cracks = predict_single_model(
                image, model_name, model_info,
                confidence_threshold, image_size, force_cpu
            )
            results.append(("Gabor Model", model_name, num_cracks, result_img))
        
        # Prepare outputs
        normal_output = None
        gabor_output = None
        
        for label, model_name, num_cracks, img in results:
            if "Normal" in label:
                normal_output = img
            else:
                gabor_output = img
        
        # Create info message
        info_parts = ["✅ **Comparison Complete!**\n"]
        info_parts.append(f"**Confidence Threshold:** {confidence_threshold}")
        info_parts.append(f"**Device:** {device}\n")
        
        for label, model_name, num_cracks, _ in results:
            info_parts.append(f"**{label}** (`{model_name}`):")
            info_parts.append(f"  - Cracks Detected: **{num_cracks}**")
        
        info = "\n".join(info_parts)
        
        print(f"\n{'='*60}")
        print("✓ Comparison complete!")
        print(f"{'='*60}\n")
        
        return normal_output, gabor_output, info
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR occurred:")
        print(error_trace)
        error_msg = f"❌ **Error during prediction:**\n```\n{str(e)}\n```\n\nCheck terminal for full traceback."
        return None, None, error_msg

def get_model_choices():
    """Get list of available model names."""
    models = get_available_models()
    if not models:
        return ["No models found"]
    return list(models.keys())

# Create Gradio interface
with gr.Blocks(title="Crack Detection System") as demo:
    gr.Markdown(
        """
        # 🔍 Crack Detection System - Model Comparison
        
        Upload an image to detect cracks using **both** trained models simultaneously.
        The system will show results from the Normal model and Gabor-enhanced model side-by-side for comparison.
        
        ⚠️ **Note:** First prediction may take 1-2 minutes while both models load into memory. Subsequent predictions will be much faster!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input Image")
            
            # Input image
            input_image = gr.Image(
                label="Upload Crack Image",
                type="numpy",
                height=500
            )
            
            gr.Markdown("### ⚙️ Detection Settings")
            
            # Confidence threshold
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.4,
                step=0.05,
                label="Confidence Threshold",
                info="Only show detections above this confidence (0.0 - 1.0)"
            )
            
            # Image size
            size_slider = gr.Slider(
                minimum=256,
                maximum=1024,
                value=512,
                step=64,
                label="Image Processing Size",
                info="Resize shortest edge to this value (larger = slower but more accurate)"
            )
            
            # Force CPU checkbox
            force_cpu_checkbox = gr.Checkbox(
                label="Force CPU Mode",
                value=False,
                info="Use CPU instead of GPU (slower but more stable)"
            )
            
            # Predict button
            predict_btn = gr.Button("🔍 Detect Cracks (Both Models)", variant="primary", size="lg")
            
            # Info text
            info_text = gr.Markdown("")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📥 Model Outputs - Side by Side Comparison")
            
            with gr.Row():
                # Normal model output
                normal_output = gr.Image(
                    label="🔹 Normal Model",
                    type="numpy",
                    height=500
                )
                
                # Gabor model output
                gabor_output = gr.Image(
                    label="🔸 Gabor Model",
                    type="numpy",
                    height=500
                )
    
    # Example images section
    gr.Markdown("### 📋 Examples")
    gr.Markdown("Try the system with sample images from the test dataset (if available)")
    
    # Get example images if they exist
    example_dir = Path("datasets/DeepCrack/test_img")
    example_images = []
    if example_dir.exists():
        example_files = sorted(list(example_dir.glob("*.jpg"))[:3])
        if example_files:
            example_images = [[str(f)] for f in example_files]
    
    if example_images:
        gr.Examples(
            examples=example_images,
            inputs=[input_image],
            label="Click an example to load it"
        )
    
    # Device info
    device_info = "🖥️ **Device:** GPU (CUDA) available ✓" if torch.cuda.is_available() else "🖥️ **Device:** CPU only (GPU not available)"
    gr.Markdown(device_info)
    
    # Event handlers
    predict_btn.click(
        fn=predict_crack_comparison,
        inputs=[input_image, confidence_slider, size_slider, force_cpu_checkbox],
        outputs=[normal_output, gabor_output, info_text]
    )
    
    gr.Markdown(
        """
        ---
        ### 📝 Instructions
        1. **Upload an image** containing cracks (concrete, pavement, etc.)
        2. **Adjust confidence threshold** if needed (lower = more detections, higher = fewer but more confident)
        3. **Click "Detect Cracks (Both Models)"** to run predictions with both models
        4. **Compare results** - Normal model vs Gabor-enhanced model side-by-side
        5. View which model detects more cracks and handles different crack types better
        
        ### 🔬 Model Comparison
        - **Normal Model**: Standard crack detection
        - **Gabor Model**: Uses Gabor filters for enhanced edge detection
        
        ### ⚙️ Model Training
        If no models are available, train both models using:
        ```bash
        # Normal model
        python tools/train_net.py --config-file configs/sparse_inst_r50_giam_crack.yaml
        
        # Gabor model
        python tools/train_net.py --config-file configs/sparse_inst_r50_giam_crack_gabor.yaml
        ```
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True to create a public link
        show_error=True
    )

