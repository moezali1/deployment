from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model=load_model('model')


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=np.array(int_features)
    col = ['Temperature', 'Oxygen', 'Humidity']
    data_unseen = pd.DataFrame([final], columns = col)
    print(int_features)
    print(final)
    prediction=predict_model(model, data=data_unseen)
    prediction=prediction.Score[0]
    output='{0:.{1}f}'.format(prediction, 2)

    if output>str(0.5):
        return render_template('index.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('index.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Score[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)