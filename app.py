from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})


# Load model & tokenizer
model_path = "./news-bias-model"
tokenizer = AutoTokenizer.from_pretrained("yashx420/news-categorizer-model")
model = AutoModelForSequenceClassification.from_pretrained("yashx420/news-categorizer-model")
model.eval()

labels = ["Center", "Left", "Right"]  # Adjust based on your training

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return jsonify({
        "prediction": labels[pred_class],
        "confidence": round(confidence, 3)
    })



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

