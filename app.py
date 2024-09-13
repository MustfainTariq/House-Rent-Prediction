from flask import Flask, redirect, render_template, request, url_for
import joblib
import pandas as pd

app = Flask(__name__)

def predict_price(location, area, bedrooms, baths):
    model = joblib.load('house_price_model.pkl')
    input_data = pd.DataFrame({
        'Location': [location],
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Baths': [baths]
    })
    predicted_price = model.predict(input_data)
    return predicted_price[0]

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    locations = ['F-8', 'DHA', 'F-6', 'D-17', 'Bahria', 'F-11', 'F-7', 'I-8', 
                 'E-7', 'F-10', 'Kuri', 'G-13', 'G-6', 'G-9', 'Shah', 'Zaraj', 
                 'E-11', 'I-10', 'Soan', 'H-13', 'Bani', 'D-12', 'G-11', 'G-10', 
                 'Gulberg', 'B-17', 'Naval', 'I-14', 'Mumtaz', 'Pakistan', 'Top', 
                 'G-15', 'Emaar', 'F-15', 'PWD', 'I-9', 'National', 'G-8', 
                 'Shehzad', 'FECHS', 'Korang', 'Margalla', 'Faisal', 'E-16', 
                 'Ghauri', 'G-7', 'G-14', 'G-12', 'Park', 'CBR', 'I-11', 
                 'Green', 'Chatha', 'G-16', 'Taramrri', 'Gulshan-e-Khudadad', 
                 'F-17', 'E-17', 'I-13', 'Tarnol', 'Khanna', 'Chak', 'Bhara', 
                 'Capital', 'Meherban', 'University', 'E-18', 'Constitution', 
                 'Tarlai', 'Police', 'Lehtarar']

    if request.method == 'POST':
        location = request.form.get('location')
        area = request.form.get('area')
        bedrooms = request.form.get('bedrooms')
        baths = request.form.get('baths')
        
        # Predict price
        predicted_price = predict_price(location, float(area), int(bedrooms), int(baths))
        
        return render_template('form.html', locations=locations, predicted_price=predicted_price)

    return render_template('form.html', locations=locations)
@app.route('/')
def home_redirect():
    return redirect(url_for('predict'))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
