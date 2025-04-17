from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Load model & tokenizer
tokenizer = BertTokenizer.from_pretrained("yashx420/news-categorizer-model")
model = BertForSequenceClassification.from_pretrained("yashx420/news-categorizer-model")
model.eval()

labels = ["Center", "Left", "Right"]  # Adjust based on your training

# === Prediction Route ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return jsonify({
        "prediction": labels[pred_class],
        "confidence": round(confidence, 3)
    })

# === Proxy News API Route ===
NEWS_API_KEY = "79e34d877d0e4b168de3864b7507ee50"

@app.route("/news", methods=["GET"])
def get_news():
    query = request.args.get("q")
    from_date = request.args.get("from", "2025-03-20")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "popularity",
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params)
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500
        
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

