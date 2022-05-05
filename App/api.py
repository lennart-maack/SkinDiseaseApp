import os
import glob
import numpy as np
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for

import prediction as pred

###################################################################################

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "image_to_predict")
GT_PATH = os.path.join("static", "predict_csv")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.route("/", methods=["GET", "POST"])
def starting_page(): 
    return render_template("home.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/multidisease", methods=["GET", "POST"])
def multidisease():
    if request.method == "POST":
        image_file = request.files["file"] # "image" -> name of the input file in index.html
        print(type(image_file))
        # make sure that image_file excists
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            print(image_location)
            image_file.save(image_location)
            return render_template("multidisease.html", prediction = "", image_loc = image_file.filename)
            
    return render_template("multidisease.html", prediction = "No photo uploaded yet")


@app.route("/prediction", methods=["GET"])
def prediction():
    
    predictor = pred.Predictor(UPLOAD_FOLDER, GT_PATH)
    prediction = predictor.predict()

    for image in os.listdir(UPLOAD_FOLDER):
        print(image)
        os.remove(os.path.join(UPLOAD_FOLDER, image))
        
    for file in os.listdir(GT_PATH):
        print(file)
        os.remove(os.path.join(GT_PATH, file))
    
    return render_template("prediction.html", prediction = np.argmax(prediction[0]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080)
