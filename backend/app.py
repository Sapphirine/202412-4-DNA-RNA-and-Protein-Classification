from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.api.classification import classify_with_model

app = Flask(__name__)
CORS(app)

@app.route('/api/classify', methods=['POST'])
def classify():
    """Handle classification requests by invoking the appropriate model."""
    data = request.json
    print(type(data))
    model = data.get('model', '')
    data.pop("model")

    result = classify_with_model(model, data)

    return jsonify({"result": result})

if __name__ == '__main__':
    """Run the Flask application on the specified host and port."""
    app.run(host='0.0.0.0', port=5001)
