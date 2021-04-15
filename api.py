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
UPLOAD_FOLDER = r"static\image_to_predict"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def starting_page(): 
    return render_template("home.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("home.html")



@app.route("/multidisease", methods=["GET", "POST"])
def multidisease():
    if request.method == "POST":
        print("got POST")
        print(request.files["file"])
        image_file = request.files["file"] # "image" -> name of the input file in index.html
        print(type(image_file))
        # make sure that image_file excists
        if image_file:
            print("got image_file")
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            return render_template("multidisease.html", prediction = "", image_loc = image_file.filename)
            
    return render_template("multidisease.html", prediction = "No photo uploaded yet")


@app.route("/prediction", methods=["GET"])
def prediction():
    
    predictor = pred.Predictor()
    prediction = predictor.predict()
        
    images = glob.glob('static/image_to_predict/*')
    for i in images:
        os.remove(i)
        
    files = glob.glob('static/predict_csv/*')
    for f in files:
        os.remove(f)
    
    return render_template("prediction.html", prediction = np.argmax(prediction[0]))


if __name__ == "__main__":
    app.run(port = 8000, debug = True)
