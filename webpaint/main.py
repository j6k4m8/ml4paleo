import base64
import io
import time

from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image

# Random Forest:
from sklearn.ensemble import RandomForestClassifier

# feature extraction:
import numpy as np


from ml4paleo.segmentation import RandomForest3DSegmenter

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data/submit", methods=["POST"])
def paint():
    # Get the image from the request JSON ("image")
    image = request.json["image"]
    # This is base64 encoded, so decode it and save as {timestamp}.png
    filename = "user_segmentation/" + str(int(time.time())) + ".png"
    # use PIL to save the image
    Image.open(io.BytesIO(base64.b64decode(image.split(",")[1]))).save(filename)

    # Train the model on X=example.jpg, y={filename}.png
    X = Image.open("example.jpg")
    y = Image.open(filename)

    # Convert the images to arrays:
    # Convert X to greyscale (just pick one of the color channels)
    X = np.array(X)[:, :, 0]
    # Convert y to seg: Flatten all color to a single value
    y = np.array(y).sum(axis=2)
    Xshape = X.shape

    # Reshape the arrays to be 1D
    X = X.reshape(-1, 1)
    y = y.reshape(-1)
    y = y > 0
    y = y.astype(int)

    # Train the model
    sample_freq = 100
    model = RandomForestClassifier(n_estimators=20, max_depth=8, n_jobs=-1)
    model.fit(X[::sample_freq], y[::sample_freq])

    # Generate a new prediction for `example.jpg`
    prediction = model.predict(X)

    # Reshape the prediction to be 2D
    prediction = prediction.reshape(Xshape)

    # Broadcast to a red image with 0 for G, B, and A channels:
    predictionimg = np.zeros((prediction.shape[0], prediction.shape[1], 4))
    predictionimg[:, :, 0] = prediction * 255
    # Set the alpha channel to 255
    predictionimg[:, :, 3] = prediction * 255

    # Save the prediction as `prediction.png`
    Image.fromarray(predictionimg.astype(np.uint8)).save("prediction.png")

    # return json of prediction in b64:
    with open("prediction.png", "rb") as f:
        return jsonify({"prediction": base64.b64encode(f.read()).decode("utf-8")})


# /api/images/next${_cachebuster}
@app.route("/api/images/next")
def next_image():
    # Send `example.jpg` as a file, using the `send_file` function
    return send_file("example.jpg", mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True)
