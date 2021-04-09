import os
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for

app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\Admin\Documents\GitHub\SkinDiseaseApp\static"

@app.route("/", methods=["GET", "POST"])
def starting_page(): 
    return render_template("home.html")

@app.route("/home", methods=["GET", "POST"])
def home_page():
    return render_template("home.html")



@app.route("/multidisease", methods=["GET", "POST"])
def multidisease():
    if request.method == "POST":
        image_file = request.files["image"] # "image" -> name of the input file in index.html
        # make sure that image_file excists
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            return render_template("multidisease.html", prediction = "You uploaded an image to get predicted")
            
    return render_template("multidisease.html", prediction = "No Image uploaded")


if __name__ == "__main__":
    app.run(port = 8000, debug = True)
