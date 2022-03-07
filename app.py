import pickle
import numpy as np
from flask import Flask, render_template, request

knn_classifier, standard_scaler = pickle.load(open('./knn_sc_diabetes.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    input_data = list()

    if request.method == 'POST':  

        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood-pressure'])
        skin_thickness = float(request.form['skin-thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        input_data += [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        input_data = np.array([input_data])
        input_data = standard_scaler.transform(input_data)

        print(input_data,type(input_data))

        my_prediction = int(knn_classifier.predict(input_data)[0])

        feature_dict = contruct_feature_dict(request.form)
              
        return render_template('result.html', diabetic=my_prediction, features=feature_dict)

if __name__ == '__main__':
    app.run(Debug=True)

def contruct_feature_dict(form_data):

    feature_dict = {}

    feature_dict['pregnancies'] = form_data['pregnancies']
    feature_dict['glucose'] = form_data['glucose']
    feature_dict['blood-pressure'] = form_data['blood-pressure']
    feature_dict['skin-thickness'] = form_data['skin-thickness']
    feature_dict['insulin'] = form_data['insulin']
    feature_dict['bmi'] = form_data['bmi']
    feature_dict['dpf'] = form_data['dpf']
    feature_dict['age'] = form_data['age']

    return feature_dict