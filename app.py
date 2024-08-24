# server.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for development

# Load the trained model
with open('LinearRegressionModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from request
        brand = data['Brand']
        type_ = data['Type']
        year_of_purchase = data['Year of Purchase']
        damage = data['Damage/Missing Parts']

        # Create DataFrame for prediction
        example_data = pd.DataFrame({
            'Brand': [brand],
            'Type': [type_],
            'Year of Purchase': [year_of_purchase],
            'Damage/Missing Parts': [damage]
        })

        # Make prediction
        prediction = model.predict(example_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'Predicted Price (INR)': prediction[0]})

if __name__ == '__main__':
    debug = True
    app.run(host='0.0.0.0', port=5000)
