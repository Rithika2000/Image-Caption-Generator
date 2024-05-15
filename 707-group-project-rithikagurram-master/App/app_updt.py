from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Using GPT2Tokenizer for language codes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Language codes for different languages
LANG_CODES = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
}

# Utility function to generate image description
def generate_description(image, language="en"):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    language_code = LANG_CODES.get(language, "english")  # Get language code from LANG_CODES
    output_ids = model.generate(pixel_values, max_length=50, num_beams=4, early_stopping=True, decoder_start_token_id=tokenizer.convert_tokens_to_ids(f"<{language_code}>"))
    description = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return description

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        image = Image.open(BytesIO(file.read()))
        language = request.form.get("language", "en")  # Get selected language from request parameters
        description = generate_description(image, language)
        return jsonify({"description": description})

if __name__ == "__main__":
    app.run(debug=True)
