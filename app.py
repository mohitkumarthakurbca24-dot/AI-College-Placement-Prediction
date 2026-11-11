import os, json, subprocess, sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "placement_model.joblib")
METRICS_PATH = os.path.join(BASE, "models", "metrics.json")

if not os.path.exists(MODEL_PATH):
    subprocess.run([sys.executable, os.path.join(BASE, "train_model.py")], check=True)

bundle = joblib.load(MODEL_PATH)
model, FEATURES = bundle["model"], bundle["features"]
with open(METRICS_PATH) as f:
    METRICS = json.load(f)

app = Flask(__name__)

LABELS = {
    "cgpa":"CGPA","tenth_percentage":"10th Percentage","twelfth_percentage":"12th Percentage",
    "backlogs":"Backlogs","programming_skill":"Programming Skill","aptitude_score":"Aptitude Score",
    "communication_score":"Communication Score","internships":"Internships",
    "certifications":"Certifications","projects":"Projects",
    "attendance_percentage":"Attendance","technical_interview_score":"Technical Interview",
    "mock_interview_score":"Mock Interview"
}

def recommendations(row):
    tips=[]
    if row["cgpa"] < 7.0: tips.append("Improve CGPA through consistent academic preparation.")
    if row["aptitude_score"] < 65: tips.append("Practice quantitative, logical and verbal aptitude.")
    if row["communication_score"] < 65: tips.append("Work on communication, presentations and group discussions.")
    if row["programming_skill"] < 6: tips.append("Strengthen programming fundamentals and problem solving.")
    if row["internships"] == 0: tips.append("Consider an internship or industry-oriented training.")
    if row["projects"] < 2: tips.append("Build at least two practical, well-documented projects.")
    if row["technical_interview_score"] < 65: tips.append("Practice technical interview questions and mock interviews.")
    if row["mock_interview_score"] < 65: tips.append("Take more mock interviews and review feedback.")
    if row["attendance_percentage"] < 75: tips.append("Improve attendance and maintain regular study habits.")
    return tips or ["Maintain your current profile and continue interview preparation."]

@app.route("/")
def home():
    return render_template("index.html", metrics=METRICS)

@app.route("/predict", methods=["POST"])
def predict():
    values={}
    for f in FEATURES:
        try:
            values[f]=float(request.form.get(f, 0))
        except ValueError:
            values[f]=0.0
    X=pd.DataFrame([values], columns=FEATURES)
    probability=float(model.predict_proba(X)[0][1])
    placed=int(probability >= 0.5)
    level = "High" if probability >= 0.75 else ("Moderate" if probability >= 0.5 else "Low")
    return render_template(
        "result.html",
        probability=round(probability*100,2),
        prediction="Likely to be Placed" if placed else "Needs Improvement",
        level=level,
        tips=recommendations(values)
    )

if __name__ == "__main__":
    app.run(debug=True)
