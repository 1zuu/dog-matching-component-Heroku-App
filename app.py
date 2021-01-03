import os
import json
import requests
import numpy as np
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from heroku_inference import InferenceModel
model = InferenceModel()

from util import *
from variables import *

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict(show_fig=False):
    message = request.get_json(force=True)
    if len(message) == 2:
        text_pad, label = get_prediction_data(message)
        model.extract_image_features(label)
        n_neighbours = model.predictions(text_pad, show_fig)
        response = {
            "neighbours": n_neighbours
                    }
        return jsonify(response)
    else:
        return "Please input both Breed and the text content"

if __name__ == "__main__": 
    app.run(debug=True, host=host, port= port, threaded=False, use_reloader=False)
