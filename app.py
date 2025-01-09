from flask import Flask, render_template, request
import os
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Azure Custom Vision parameters
prediction_key = '34hXL6V6Dnus0zudNhEW1VwTGRVmqNtZrSpz3OyHbR4Nhhy80lkgJQQJ99BAACYeBjFXJ3w3AAAIACOGnHOn'
prediction_endpoint = 'https://aicustomvisionclassifyimg001-Prediction.cognitiveservices.azure.com/'
project_id = 'ee4cb63f-8e65-48cb-9900-2a3171335b98'
published_model_name = 'AnimalSpeciesIdentification'

# Set up prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(prediction_endpoint, credentials)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    img_base64 = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        # Convert the image to base64 for display
        image_data = file.read()
        img = Image.open(BytesIO(image_data))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Make prediction
        prediction = predictor.classify_image(project_id, published_model_name, image_data)

        # Process prediction result
        results = [(tag.tag_name, tag.probability) for tag in prediction.predictions]

    return render_template('index.html', results=results, image_data=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
