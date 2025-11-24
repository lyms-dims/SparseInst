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

def predict_crack(image, model_name, confidence_threshold, image_size, force_cpu):
    """Predict cracks on an input image."""
    try:
        print(f"\n{'='*50}")
        print(f"Starting prediction...")
        print(f"Model: {model_name}")
        print(f"Confidence: {confidence_threshold}")
        print(f"Image size: {image_size}")
        print(f"Force CPU: {force_cpu}")
        
        if image is None:
            print("Error: No image provided")
            return None, "Please upload an image first."
        
        print(f"Image shape: {image.shape}")
        
        # Get available models
        available_models = get_available_models()
        
        if not available_models:
            print("Error: No models found")
            return None, "❌ No trained models found in 'output/' directory. Please train a model first."
        
        if model_name not in available_models:
            print(f"Error: Model {model_name} not in available models")
            return None, f"❌ Model '{model_name}' not found."
        
        # Get model info
        model_info = available_models[model_name]
        print(f"Model weights: {model_info['weights']}")
        print(f"Model config: {model_info['config']}")
        
        # Check if files exist
        if not os.path.exists(model_info['weights']):
            return None, f"❌ Model weights file not found: {model_info['weights']}"
        if not os.path.exists(model_info['config']):
            return None, f"❌ Config file not found: {model_info['config']}"
        
        # Check model file size
        model_size_mb = os.path.getsize(model_info['weights']) / (1024 * 1024)
        print(f"Model file size: {model_size_mb:.1f} MB")
        
        # Use cache key with device info
        device_suffix = "_cpu" if force_cpu else "_gpu"
        cache_key = model_name + device_suffix
        
        # Load or get cached predictor
        if cache_key not in predictors_cache:
            print(f"Loading model for the first time...")
            print(f"This may take 30-60 seconds on first run...")
            cfg = setup_cfg(
                model_info["config"],
                model_info["weights"],
                confidence_threshold,
                image_size,
                force_cpu=force_cpu
            )
            print(f"Config created, device: {cfg.MODEL.DEVICE}")
            print(f"Loading model weights from: {model_info['weights']}")
            print(f"Creating predictor (this is the slow part)...")
            sys.stdout.flush()  # Force output to show immediately
            
            predictors_cache[cache_key] = DefaultPredictor(cfg)
            
            print("✓ Predictor created and cached!")
            print(f"✓ Model loaded successfully on {cfg.MODEL.DEVICE}!")
        else:
            print("Using cached predictor")
        
        predictor = predictors_cache[cache_key]
        
        # Update confidence threshold if needed
        predictor.cfg.defrost()
        predictor.cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = confidence_threshold
        predictor.cfg.INPUT.MIN_SIZE_TEST = image_size
        predictor.cfg.freeze()
        
        # Convert image to BGR for OpenCV
        print("Converting image format...")
        if len(image.shape) == 2:
            # Grayscale
            im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            # RGB to BGR
            im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        print(f"Running detection on image of shape {im.shape}...")
        # Run detection
        outputs = predictor(im)
        print("Detection complete!")
        
        # Get predictions
        instances = outputs["instances"].to("cpu")
        num_instances = len(instances)
        print(f"Found {num_instances} crack instances")
        
        # Visualize
        print("Creating visualization...")
        metadata = MetadataCatalog.get("deepcracks_val")
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(instances)
        
        # Convert back to RGB
        result_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        print("Visualization complete!")
        
        # Create info message
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        info = f"""
✅ **Detection Complete!**
- **Cracks Detected:** {num_instances}
- **Model:** {model_name}
- **Confidence Threshold:** {confidence_threshold}
- **Device:** {device}
        """
        
        print(f"{'='*50}\n")
        return result_image, info
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR occurred:")
        print(error_trace)
        error_msg = f"❌ **Error during prediction:**\n```\n{str(e)}\n```\n\nCheck terminal for full traceback."
        return None, error_msg

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
        # 🔍 Crack Detection System
        
        Upload an image to detect cracks using trained deep learning models.
        The system will highlight detected cracks and provide instance segmentation.
        
        ⚠️ **Note:** First prediction may take 30-60 seconds while the model loads into memory. Subsequent predictions will be much faster!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            
            # Input image
            input_image = gr.Image(
                label="Upload Image",
                type="numpy",
                height=400
            )
            
            # Model selection
            model_dropdown = gr.Dropdown(
                choices=get_model_choices(),
                label="Select Model",
                value=get_model_choices()[0] if get_model_choices()[0] != "No models found" else None,
                info="Choose a trained model from the output directory"
            )
            
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
            predict_btn = gr.Button("🔍 Detect Cracks", variant="primary", size="lg")
            
            # Refresh models button
            refresh_btn = gr.Button("🔄 Refresh Model List", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Output")
            
            # Output image
            output_image = gr.Image(
                label="Detected Cracks",
                type="numpy",
                height=400
            )
            
            # Info text
            info_text = gr.Markdown("")
    
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
        fn=predict_crack,
        inputs=[input_image, model_dropdown, confidence_slider, size_slider, force_cpu_checkbox],
        outputs=[output_image, info_text]
    )
    
    def refresh_models():
        choices = get_model_choices()
        return gr.Dropdown(choices=choices, value=choices[0] if choices[0] != "No models found" else None)
    
    refresh_btn.click(
        fn=refresh_models,
        outputs=[model_dropdown]
    )
    
    gr.Markdown(
        """
        ---
        ### 📝 Instructions
        1. **Upload an image** containing cracks (concrete, pavement, etc.)
        2. **Select a trained model** from the dropdown menu
        3. **Adjust confidence threshold** if needed (lower = more detections, higher = fewer but more confident)
        4. **Click "Detect Cracks"** to run the prediction
        5. View the results with highlighted crack instances
        
        ### ⚙️ Model Training
        If no models are available, train a model first using:
        ```bash
        python tools/train_net.py --config-file configs/sparse_inst_r50_giam_crack.yaml
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

