import logging

from flask import Flask, Response, jsonify, request

from model.model_wrapper import ModelWrapper

logger = logging.getLogger("prod")
logger.setLevel(logging.DEBUG)

model = ModelWrapper.deserialize("model/model_wrapper.npz")
logger.info("ModelWrapper deserialized")
model.warmup()
logger.info("Model warmed up")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    try:
        data = request.get_json(force=True)
        title = data["title"]
        description = data["description"]
        categories = data["categories"]
        duration = data["duration"]

        logger.info("Processing request with title: %s...", title[:30])

        prediction = model.predict(title, description, categories, duration)
        logger.info("Prediction: %s", prediction)

        return jsonify(
            {
                "prediction": prediction,
                "version": "v2.0.0",
            },
        )
    except Exception as exception:
        logger.exception("Error during prediction: %s", exception)
        return jsonify(
            {
                "error": str(exception),
            },
        ), 400


if __name__ == "__main__":
    from waitress import serve

    logger.info("Server started on port 5000")
    serve(app, host="0.0.0.0", port=5000)
