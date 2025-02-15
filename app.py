from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model("models/disease_detection_model.h5")

# Categories for diseases
CATEGORIES = [
    "Brain_Tumor_Glioma", "Brain_Tumor_Meningioma", "Brain_Tumor_Normal",
    "Brain_Tumor_Pituitary", "COVID", "Glaucoma", "Lung_Opacity", 
    "Macular_Degeneration", "Normal_Chest", "Normal_Eye", 
    "Retina_Disease", "Skin_Cancer_Benign", "Skin_Cancer_Malignant", 
    "Viral_Pneumonia"
]
IMG_SIZE = 128

# Load health information from Excel file
health_info_df = pd.read_excel("solutions/Normal_Conditions_Health_Info.xlsx")

# Map diseases to their respective solutions
def get_solution(disease):
    row = health_info_df.loc[health_info_df['Category'] == disease]
    if not row.empty:
        return {
            "Description": row.iloc[0]['Description'],
            "Prevention": row.iloc[0]['Prevention'],
            "Medicines": row.iloc[0]['Medicines'],
            "What_to_Avoid": row.iloc[0]['What to Avoid'],
            "What_to_Eat": row.iloc[0]['What to Eat'],
            "Ayurvedic": row.iloc[0]['Ayurvedic Medicine and Practices'],
            "Surgery": row.iloc[0]['Surgery/Operation Required']
        }
    return {"error": "No solution found for this disease"}

# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the AI Disease Detection and Solution API!"})

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        # Preprocess the uploaded image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

        # Predict disease
        prediction = model.predict(img)
        predicted_disease = CATEGORIES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # Fetch solutions for the predicted disease
        solution = get_solution(predicted_disease)

        return jsonify({
            "prediction": predicted_disease,
            "confidence": confidence,
            "solution": solution
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
