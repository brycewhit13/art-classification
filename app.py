import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import torch
from torchvision.models import resnet50
from PIL import Image

####################
# Helper Functions #
####################

def load_trained_model():
    # Get the model path
    model_path = os.path.join("models", "model.pth")
    
    # Load the model
    model = resnet50()
    # Replace the output layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode and return it
    model.eval()
    return model

def make_prediction(trained_model, image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Resize and format the image
    image = image.resize((224, 224))
    image = np.moveaxis(np.array(image), -1, 0)
    image = torch.tensor(image).float()
    image = image.unsqueeze(0)
    
    # Pass the image through the model
    output = trained_model(image)
    pred = torch.argmax(output)
    
    # Convert the prediction to a string and return it
    if pred == 0:
        return "Human"
    return "AI"

def make_proba_prediction(trained_model, image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Resize and format the image
    image = image.resize((224, 224))
    image = np.moveaxis(np.array(image), -1, 0)
    image = torch.tensor(image).float()
    image = image.unsqueeze(0)
    
    # Pass the image through the model
    output = trained_model(image)
    probas = torch.nn.functional.softmax(output, dim=1)
    
    # Return the probability of the image being AI and human
    output_dict = {"Human": np.round(probas[0][0].item(), 2), "AI": np.round(probas[0][1].item(), 2)}
    return output_dict


#############
# Flask App #
#############

# Initialize the app
app = Flask(__name__)

# Load the trained model
trained_model = load_trained_model()

# R
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route to handle the image upload and make predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    # Make the predictions
    pred = make_prediction(trained_model, file)
    pred_probas = make_proba_prediction(trained_model, file)

    # Return the results as a response
    result = {
        "prediction": pred,
        "prediction_probabilities": pred_probas
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

#if __name__ == "__main__":
    # #app.run(host='127.0.0.1', port=5000, debug=True)
    # # Load the trained model and image
    # trained_model = load_trained_model()
    # #image_path = '/Users/brunovalan/Desktop/valan_headshot.jpg'
    # image_path = os.path.join('data', 'ai_art_classification', 'train', 'AI_GENERATED', '1.jpg')
    
    # # Make the predictions
    # pred = make_prediction(trained_model, image_path)
    # pred_probas = make_proba_prediction(trained_model, image_path)
    
    # # Print the results
    # print(f"Prediction: {pred}")
    # print(f"Prediction Probabilities: {pred_probas}")