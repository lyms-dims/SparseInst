# Crack Detection GUI - Model Comparison

A user-friendly web interface for detecting cracks in images using **both** trained deep learning models simultaneously. Compare Normal vs Gabor-enhanced models side-by-side!

## 🚀 Quick Start

### 1. Install Gradio

If you haven't already, install Gradio:

```bash
pip install gradio
```

### 2. Run the GUI

```bash
python app.py
```

The GUI will start and automatically open in your browser at `http://localhost:7860`

## 📖 Usage Guide

### Step-by-Step Instructions

1. **Launch the Application**
   ```bash
   python app.py
   ```

2. **Upload an Image**
   - Click on the "Upload Image" box
   - Select an image containing cracks (supports .jpg, .png, .bmp, etc.)
   - Or drag and drop an image directly

3. **Configure Detection Settings**
   - **Confidence Threshold**: Adjust from 0.0 to 1.0 (default: 0.4)
     - Lower values = more detections (may include false positives)
     - Higher values = fewer detections (only high-confidence cracks)
   - **Image Processing Size**: Resize shortest edge (default: 512)
     - Smaller = faster but less accurate
     - Larger = slower but more accurate
   - **Force CPU Mode**: Check this if GPU hangs or isn't available

4. **Run Detection**
   - Click the "🔍 Detect Cracks (Both Models)" button
   - Wait for processing (1-2 minutes first time, then faster)
   - View the results from both models side-by-side

5. **Compare Results**
   - **Left panel**: Normal model predictions
   - **Right panel**: Gabor model predictions
   - Check the info panel for:
     - Number of cracks detected by each model
     - Confidence threshold applied
     - Processing device (GPU/CPU)
   - See which model performs better on your specific crack type!

## 🎯 Features

- **Easy to Use**: Simple drag-and-drop interface
- **Dual Model Comparison**: Automatically runs both Normal and Gabor models
- **Side-by-Side Results**: Compare model outputs instantly
- **Real-time Detection**: Get predictions in seconds (after initial load)
- **Adjustable Settings**: Fine-tune confidence threshold and image size
- **GPU Acceleration**: Automatically uses CUDA if available (with CPU fallback)
- **Example Images**: Quick test with pre-loaded samples
- **Model Caching**: Models stay loaded for fast subsequent predictions

## 🔧 Advanced Options

### Sharing the Interface

To create a public link that others can access:

Edit `app.py` and change:
```python
demo.launch(share=True)  # Creates a public shareable link
```

### Custom Port

Change the port number:
```python
demo.launch(server_port=8080)  # Use port 8080 instead
```

### Running on a Server

For deployment on a remote server:
```bash
python app.py
```
Access via `http://your-server-ip:7860`

## 📦 Model Requirements

The GUI automatically detects trained models in the `output/` directory and runs **both** models for comparison. You should have:
- **Normal Model**: Standard crack detection model
- **Gabor Model**: Gabor-enhanced crack detection model

Each model should have:
- `model_final.pth` - The trained weights
- `config.yaml` - Model configuration (optional, will auto-detect)

### Training Both Models

For full comparison functionality, train both models:

```bash
# Train standard model (required)
python tools/train_net.py --config-file configs/sparse_inst_r50_giam_crack.yaml

# Train Gabor model (required)
python tools/train_net.py --config-file configs/sparse_inst_r50_giam_crack_gabor.yaml
```

After training, both models will be detected automatically. If only one model is available, the GUI will show only that model's output.

## 🐛 Troubleshooting

### "No models found"
- Ensure you have trained models in the `output/` directory
- Each model folder should contain `model_final.pth`
- Click "🔄 Refresh Model List" after training

### GPU not detected
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- The system will automatically fall back to CPU

### Import errors
- Ensure all dependencies are installed
- Run from the project root directory
- Check that `sparseinst` module is available

### Slow processing
- Reduce "Image Processing Size" to 256 or 384
- Use GPU if available
- Consider using a smaller model

## 🎨 Customization

### Adding More Examples

Add your own example images:
```python
example_images = [
    ["path/to/image1.jpg"],
    ["path/to/image2.jpg"],
]
```

## 📊 Comparison with CLI

| Feature | GUI (`app.py`) | CLI (`predict.py`) |
|---------|----------------|-------------------|
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Batch Processing | ❌ | ✅ |
| Model Comparison | ✅ Side-by-side | ✅ Separate folders |
| Visual Feedback | ✅ Real-time | ✅ Saved files |
| Adjustable Settings | ✅ Interactive sliders | ✅ CLI args |
| Best For | Single images, demos, comparison | Batch processing, automation |

## 💡 Tips

- **For best results**: Use confidence threshold between 0.3-0.5
- **For speed**: Reduce image size to 384 or 256
- **For accuracy**: Increase image size to 768 or 1024 (requires more memory)
- **Multiple detections**: Try lowering confidence threshold
- **Clean results**: Try raising confidence threshold

## 🔗 Related Tools

- `tools/predict.py` - Command-line batch processing
- `tools/train_net.py` - Model training
- `tools/visualize_json_results.py` - Visualization utilities

---

**Enjoy crack detection! 🎉**

