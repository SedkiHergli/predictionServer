import pandas as pd
import numpy as np
from scipy import *
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from numpy.random import seed
import keras
import numpy as np
from keras.models import load_model

from flask import Flask, jsonify
from flask import request, after_this_request

import json


app = Flask(__name__)

@app.route('/input', methods=['POST'])
def post_data():
    try :
        model = load_model("model.model")
        dateTime = datetime.datetime.now()
        dateTime = pd.to_datetime(dateTime, format='%m/%d/%Y %I:%M:%S %p')
        hour = dateTime.hour
        dayofWeek = dateTime.dayofweek
        month= dateTime.month
        year= dateTime.year
        day= dateTime.day
        #["hour", "Day", "Month", "Year", "Day of Week", "Latitude", "Longitude"]
        a=[[hour, day, month, year, dayofWeek, request.get_json()["latitude"], request.get_json()["longitude"]]]
        label=['ASSAULT', 'BATTERY', 'BURGLARY', 'HOMICIDE', 'INTIMIDATION',
            'KIDNAPPING', 'MOTOR VEHICLE THEFT', 'OBSCENITY', 'OTHER OFFENSE',
            'PUBLIC PEACE VIOLATION', 'ROBBERY', 'SEX OFFENSE', 'THEFT',
            'WEAPONS VIOLATION']
        y_pred = model.predict_classes(np.array(a), batch_size=32, verbose=0)
        y_p = model.predict(np.array(a), batch_size=32, verbose=0)
        l=[]
        for i in range(len(y_p[0])):
            l.append({label[i]:str("{0:.2f}".format(y_p[0][i]*100))})
        dict_output = {"result" : label[y_pred[0]],"percentage":l}
        json_final= {"result":dict_output}
        jsouna = json_final
        jsouna=json.dumps(jsouna,separators=(',',':'))
        keras.backend.clear_session()
    except :
        state_sep = "invalid"
        jsouna=json.dumps({"result":state_sep,})

    print("jsouna ! ",jsouna)
    return jsouna



if __name__ == '__main__':
    app.run(debug=True)