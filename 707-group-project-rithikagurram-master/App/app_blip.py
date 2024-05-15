from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

app = Flask(__name__)

# Load the pre-trained BLIP model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Utility function to generate image description
def generate_description(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=50, num_beams=4, early_stopping=True)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption

@app.route("/generate_description", methods=["GET"])
def generate_description_route():
    image_url = request.args.get("image_url")
    description = generate_description(image_url)
    return jsonify({"description": description})

if __name__ == "__main__":
    app.run(debug=True)