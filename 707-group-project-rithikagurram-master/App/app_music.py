from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from gtts import gTTS
import os

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Utility function to generate image description
def generate_description(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=50, num_beams=4, early_stopping=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Function to generate audio from text
def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "caption_audio.mp3"
    tts.save(audio_file)
    return audio_file

# Flask route to generate description and audio
@app.route("/generate_description", methods=["GET"])
def generate_description_route():
    image_url = request.args.get("image_url")
    description = generate_description(image_url)
    audio_file = generate_audio(description)
    return jsonify({"description": description, "audio_file": audio_file})

if __name__ == "__main__":
    app.run(debug=True)