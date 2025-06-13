from flask import Flask, request, jsonify
import re, joblib
from textblob import TextBlob
app = Flask(__name__)

@app.route("/")
def index():
    return "Attrition Risk API — send POST /predict {\"feedback\": \"...\"}"


model = joblib.load("models/model.pkl")
le = joblib.load("models/label_encoder.pkl")

def rule_based_strategy(feedback):
    text = feedback.lower()
    if "workload" in text or "overwhelming" in text:
        return "Review workloads; promote work-life balance; offer support."
    if "recognition" in text or "unnoticed" in text:
        return "Increase recognition and manager feedback; implement regular check-ins."
    if "career" in text or "advancement" in text:
        return "Create clear career pathways; offer mentorship and growth opportunities."
    if "communication" in text:
        return "Improve leadership communication; increase transparency and consistency."
    if "resources" in text or "technology" in text:
        return "Invest in updated tools; provide resource support and training."
    return "Conduct stay interviews to understand concerns."

def predict(feedback_text: str):
    sentiment = TextBlob(feedback_text).sentiment.polarity
    from datetime import datetime
    year = datetime.now().year
    x = [[year, sentiment]]
    prob = model.predict_proba(x)[0][1]
    at_risk = bool(prob >= 0.5)

    if sentiment <= 0.1:
        at_risk = True

    strategy = rule_based_strategy(feedback_text) if at_risk else \
               "Maintain current engagement and recognition programs."

    return {
        "sentiment_score": round(sentiment, 3),
        "model_probability": round(prob, 3),
        "predicted_attrition_risk": at_risk,
        "engagement_strategy": strategy
    }


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    feedback = data.get("feedback", "")
    if not feedback:
        return jsonify({"error": "Missing 'feedback' field"}), 400
    result = predict(feedback)
    result["feedback"] = feedback
    return jsonify(result)

if __name__ == "__main__":
    app.run()      
