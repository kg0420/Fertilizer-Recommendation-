from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load model and encoders
with open("fertilizer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Fertilizer recommendation.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        temp = int(request.form['Temparature'])
        humidity = int(request.form['Humidity '])  # no space, strip in CSV
        moisture = int(request.form['Moisture'])
        soil_type = request.form['SoilType']
        crop_type = request.form['CropType']
        nitrogen = int(request.form['Nitrogen'])
        potassium = int(request.form['Potassium'])
        phosphorous = int(request.form['Phosphorous'])

        # Encode categorical fields
        soil_encoded = label_encoders['Soil Type'].transform([soil_type])[0]
        crop_encoded = label_encoders['Crop Type'].transform([crop_type])[0]

        # Prepare input
        input_data = pd.DataFrame([{
            'Temparature': temp,
            'Humidity ': humidity,
            'Moisture': moisture,
            'Soil Type': soil_encoded,
            'Crop Type': crop_encoded,
            'Nitrogen': nitrogen,
            'Potassium': potassium,
            'Phosphorous': phosphorous
        }])

        # Predict probabilities
        proba = model.predict_proba(input_data)[0]

        # Get top 3 fertilizers
        top_k = 3
        top_indices = proba.argsort()[-top_k:][::-1]  # top K indices
        top_fertilizers = label_encoders['Fertilizer Name'].inverse_transform(top_indices)
        top_probs = proba[top_indices]

        result_text = "Top Recommended Fertilizers: "
        for fert, prob in zip(top_fertilizers, top_probs):
            result_text += f"- {fert} (Confidence: {prob:.2f})  "

        return render_template('Fertilizer recommendation.html', prediction_text=result_text)

    except Exception as e:
        return render_template('Fertilizer recommendation.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
