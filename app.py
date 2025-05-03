from flask import Flask, request, jsonify
import os
import base64
import uuid
from helpers.ai_service import predict_disease
from helpers.gpt_helper import get_cause_from_gpt, get_remedy_from_gpt
from helpers.weather_utils import get_manual_weather

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def save_base64_image(base64_str):
    image_data = base64.b64decode(base64_str.split(",")[-1])
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_path


@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Accept file or base64 image
    if 'image' in request.files:
        image = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(filepath)
    elif 'image_base64' in request.json:
        filepath = save_base64_image(request.json['image_base64'])
    else:
        return jsonify({"error": "No image found"}), 400

    # Get location & weather input
    location = request.form.get('location') or request.json.get('location')
    weather_data = get_manual_weather(location)  # placeholder

    # Predict from AI model
    predicted_pest = predict_disease(filepath)

    # Get cause from GPT
    cause = get_cause_from_gpt(predicted_pest, weather_data)

    # Get remedy from GPT
    remedy = get_remedy_from_gpt(predicted_pest, cause)

    return jsonify({
        "pest": predicted_pest,
        "weather": weather_data,
        "cause": cause,
        "remedy": remedy
    })


if __name__ == '__main__':
    app.run(debug=True)
