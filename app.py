import io
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras_core.models import load_model # Nueva forma de importar modelos

app = Flask(__name__)
CORS(app)

MODEL_ID = "alexanderkroner/MSI-Net"
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading model {MODEL_ID}...")
        try:
            _model = from_pretrained_keras(MODEL_ID)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e
    return _model

@app.route('/analizar-inercia', methods=['POST'])
def analizar_inercia():
    try:
        model = get_model()
    except Exception as e:
        return jsonify({'error': 'Model could not be loaded on server start.'}), 500

    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({'error': 'Invalid input. JSON with "image_url" required.'}), 400

    image_url = data['image_url']

    # 1. Download Image
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        # Open image and convert to RGB
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Failed to download or process image: {str(e)}'}), 400

    original_size = img.size # (Width, Height)

    # 2. Preprocess
    # Resize to model input shape (320, 240) - (Width, Height) for PIL
    # Model expects (Batch, Height, Width, Channels) i.e., (1, 240, 320, 3)
    target_width = 320
    target_height = 240
    
    img_resized = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    
    x = np.array(img_resized, dtype=np.float32)
    
    # Convert RGB to BGR (ImageNet standard for Caffe models, often used in Keras ports)
    x = x[..., ::-1]
    
    # Subtract ImageNet mean
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    
    # Add batch dimension
    x = np.expand_dims(x, axis=0) # Shape: (1, 240, 320, 3)

    # 3. Predict
    # Result shape is typically (1, 240, 320, 1) usually
    try:
        preds = model.predict(x, verbose=0)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Extract 2D map
    saliency_map = preds[0, :, :, 0] # (240, 320)

    # 4. Post-process
    # Resize saliency map back to original image size
    # We use PIL for high-quality resizing
    saliency_pil = Image.fromarray(saliency_map)
    saliency_pil = saliency_pil.resize(original_size, Image.Resampling.BILINEAR)
    saliency_data = np.array(saliency_pil)

    # Normalize to 0-255
    min_val = np.min(saliency_data)
    max_val = np.max(saliency_data)

    if max_val - min_val > 1e-5:
        saliency_data = (saliency_data - min_val) / (max_val - min_val)
        saliency_data = saliency_data * 255.0
    else:
        # If constant, set to 0 (or keeping it constant scaled)
        saliency_data = np.zeros_like(saliency_data)

    heatmap_data = saliency_data.astype(np.uint8).tolist()

    return jsonify({
        'heatmap_data': heatmap_data
    })

if __name__ == '__main__':
    # Load model immediately on start
    try:
        get_model()
    except:
        pass # Will retry on request
    app.run(debug=True, host='0.0.0.0', port=5000)

