from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])
        Year = int(request.form['Year'])
        Years_of_Service = 2026 - Year
        Fuel_Type_Diesel = 1 if request.form['Fuel_Type'] == 'Diesel' else 0
        Fuel_Type_Petrol = 1 if request.form['Fuel_Type'] == 'Petrol' else 0
        Seller_Type_Individual = 1 if request.form['Seller_Type'] == 'Individual' else 0
        Transmission_Manual = 1 if request.form['Transmission'] == 'Manual' else 0

        features = [np.array([Present_Price, Kms_Driven, Owner, Years_of_Service, 
                              Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual])]

        prediction = model.predict(features)
        return render_template('index.html', prediction_text=f'Estimated Selling Price: ₹ {round(prediction[0], 2)} Lakhs')

if __name__ == "__main__":
    app.run(debug=False)
