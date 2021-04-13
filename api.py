import os
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for

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
def home_page():
    return render_template("home.html")



@app.route("/multidisease", methods=["GET", "POST"])
def multidisease():
    if request.method == "POST":
        print("got POST")
        print(request.files["file"])
        image_file = request.files["file"] # "image" -> name of the input file in index.html
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


if __name__ == "__main__":
    app.run(port = 8000, debug = True)
