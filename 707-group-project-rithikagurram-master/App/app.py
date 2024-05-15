from flask import Flask, request, jsonify, send_file
import requests
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Utility function to generate image description
def generate_description(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=50, num_beams=4, early_stopping=True)
    description = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return description

@app.route("/generate_description", methods=["POST"])
def generate_description_route():
    # Check if image is uploaded or a URL is provided
    if "image_file" in request.files:
        # If image is uploaded, read and process it
        image_file = request.files["image_file"]
        image = Image.open(BytesIO(image_file.read()))
    elif "image_url" in request.form:
        # If URL is provided, fetch and process the image
        image_url = request.form["image_url"]
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    else:
        return jsonify({"error": "No image provided"}), 400

    description = generate_description(image)
    return jsonify({"description": description})

@app.route("/upload_image", methods=["POST"])
def upload_image():
    # Check if the POST request has the file part
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        image = Image.open(BytesIO(file.read()))
        description = generate_description(image)
        return jsonify({"description": description})

if __name__ == "__main__":
    app.run(debug=True)
