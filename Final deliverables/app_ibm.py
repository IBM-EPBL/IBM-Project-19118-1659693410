import requests
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__,template_folder='templates')
API_KEY = "XLRNkMl2L6Dz9nQ2xxbnVnISiD9KtcKgL1ioS1uUO30O"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

@app.route('/')
def index():
 return render_template('index.html')


@app.route("/predict")
def predict():
 return render_template('prediction.html')

@app.route('/y_predict', methods=['GET', 'POST'])
def y_predict():
 Fuel_Type_Diesel = 0
 regyear = int(request.form['regyear'])
 powerps = float(request.form['powerps'])
 kms = float(request.form['kms'])
 regmonth = int(request.form.get('regmonth'))
 gearbox = request.form['gearbox']
 damage = request.form['dam']
 model = request.form.get('modeltype')
 brand = request.form.get('brand')
 fuelType = request.form.get('fuel')
 vehicletype = request.form.get('vehicletype')
 new_row = {'yearOfRegistration': regyear, 'powerPS': powerps, 'kilometer': kms,
            'monthOfRegistration': regmonth, 'gearbox': gearbox, 'notRepairedDamage': damage,
            'model': model, 'brand': brand, 'fuelType': fuelType,
            'vehicleType': vehicletype}
 print(new_row)
 new_df = pd.DataFrame(columns=['vehicleType', 'yearOfRegistration', 'gearbox',
                                'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType',
                                'brand', 'notRepairedDamage'])
 new_df = new_df.append(new_row, ignore_index=True)
 labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
 mapper = {}
 for i in labels:
     mapper[i] = LabelEncoder()
     mapper[i].classes_ = np.load(str('classes' + i + '.npy'), allow_pickle=True)
     tr = mapper[i].fit_transform(new_df[i])
     new_df.loc[:, i + '_labels'] = pd.Series(tr, index=new_df.index)
 labeled = new_df[['yearOfRegistration'
                      , 'powerPS'
                      , 'kilometer'
                      , 'monthOfRegistration'
                   ]
                  + [x + '_labels' for x in labels]]
 X = labeled.values
 print(X)

 payload_scoring = {"input_data": [{"field": [['yearOfRegistration', 'powerPS', 'kilometer',
            'monthOfRegistration', 'gearbox', 'notRepairedDamage',
            'model', 'brand', 'fuelType',
            'vehicleType']], "values": X.tolist()}]}
 response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/4965bcba-59dc-4c55-8155-c45be9923199/predictions?version=2022-11-16',
   json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
 pred = response_scoring.json()
 print(pred)
 out = pred['predictions'][0]['values'][0][0]
 return render_template('prediction.html', ypred='The resale value predicted is ${:.2f}'.format(out))




if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=False)