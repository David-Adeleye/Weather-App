from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load the saved model
with open('naive_bayes_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the saved LabelEncoder
with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    temp_min = float(request.form['temp_min'])
    temp_max = float(request.form['temp_max'])
    precipitation = float(request.form['precipitation'])
    wind = float(request.form['wind'])

    # Create a new data point with feature names
    new_data_point = [[temp_min, temp_max, precipitation, wind]]
    feature_names = ['temp_min', 'temp_max', 'precipitation', 'wind']
    new_data_point_with_names = pd.DataFrame(new_data_point, columns=feature_names)

    # Make predictions on the new data point
    predicted_label = loaded_model.predict(new_data_point_with_names)

    # Perform inverse transformation using the LabelEncoder
    original_label = le.inverse_transform(predicted_label)

    # Render the prediction result
    return render_template('result.html', prediction=original_label[0])

if __name__ == '__main__':
    app.run(debug=True)
