import os
from flask import Flask, render_template
import torch
from torchvision import resnet50

####################
# Helper Functions #
####################

def load_trained_model():
    # Get the model path
    model_path = os.path.join("models", "model.pt")
    
    # Load the model
    model = resnet50()
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode and return it
    model.eval()
    return model

def make_prediction(trained_model, image):
    # Pass the image through the model
    output = trained_model(image.float())
    pred = torch.argmax(output)
    
    # Convert the prediction to a string and return it
    if pred == 0:
        return "Human"
    return "AI"

def make_proba_prediction(trained_model, image):
    # Pass the image through the model
    output = trained_model(image.float())
    probas = torch.nn.functional.softmax(output, dim=1)
    
    # Return the probability of the image being AI and human
    output_dict = {"Human": probas[0][0].item(), "AI": probas[0][1].item()}
    return output_dict


#############
# Flask App #
#############

# Initialize the app
app = Flask(__name__)

# Path for user to be able to upload an image
@app.route("/", methods=["GET", "POST"])
def index():
    raise NotImplementedError()

# Path for the model to make a prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    raise NotImplementedError()


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)