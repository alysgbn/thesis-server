
from flask import Flask, request, jsonify, url_for
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import numpy as np
import werkzeug
import math
import os

app = Flask(__name__)

# Import models 
model = tf.keras.models.load_model('models/best_finetuned_model.h5')
unet_model = tf.keras.models.load_model('models/unet_efficientnetv2_casia.h5')
# Define a list of class labels/names
class_names = ['Tampered', 'Authentic']

# For Detection
image_size = (128, 128)
def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

# For UNET 
unet_image_size = (224,224)
def prepare_ela_image(image_path):
    ela_image = convert_to_ela_image(image_path, 90)
    ela_image_resized = ela_image.resize(unet_image_size)
    ela_array = np.array(ela_image_resized) / 255.0
    return ela_array

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image foundsfds'})

    image = request.files['image']
    filename = werkzeug.utils.secure_filename(image.filename)
    image.save("./uploadedimages/" + image.filename)
    return jsonify({'image_name': image.filename})

    
@app.route('/classify', methods=['POST'])
def classify():
    # Check if the request contains an image
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_path = request.files['image']
    
    # Load the image and preprocess it
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    
    # Make predictions
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis = 1)[0]
    predicted_class_name = class_names[predicted_class_index]

    print(predicted_class_name)
    confidence = np.amax(predictions) * 100
    print(confidence)
    final_predictions = (str(math.trunc(confidence)) + "% " + predicted_class_name)
    if predicted_class_name == 'Authentic':
        return jsonify({'class_label': predicted_class_name})
    else:

        ela_image = prepare_ela_image(image_path)
        test_image_expanded = np.expand_dims(ela_image, axis=0)
        prediction = unet_model.predict(test_image_expanded)
        return jsonify({'class_label': prediction})


# Function to perform Error Level Analysis (ELA) preprocessing
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    # Open image and convert to RGB mode
    image = Image.open(path).convert('RGB')

    # Save the image with specified quality
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    # Calculate ELA by taking the absolute difference between original and recompressed image
    ela_image = ImageChops.difference(image, temp_image)
    # Calculate scale for enhancing brightness based on pixel extremas
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 350 / max_diff
    
    # Enhance brightness with calculated scale
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    os.remove(temp_filename)
    return ela_image


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(debug=True, host='0.0.0.0', port=port)


