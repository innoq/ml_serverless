# 
#curl "https://us-central1-mlperlin-224515.cloudfunctions.net/predict_diabetes?age=0.038076&sex=0.050680&bmi=0.061696&bp=0.021872&s1=-0.044223&s2=-0.034821&s3=-0.043401&s4=-0.002592&s5=0.019908&s6=-0.017646"

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import abort
import json

estimatorMap = {}

def predict_diabetes(request):
    args = request.args
    if(request.args and 'age' in args and 'age' in args and 'sex' in args and 'bmi' in args and 'bp' in args 
    and 's1' in args and 's2' in args and 's3' in args and 's3' in args and 's5' in args and 's6' in args):
        age = float(request.args.get('age'))
        sex = float(request.args.get('sex'))
        bmi = float(request.args.get('bmi'))
        bp = float(request.args.get('bp'))
        s1 = float(request.args.get('s1'))
        s2 = float(request.args.get('s2'))
        s3 = float(request.args.get('s3'))
        s4 = float(request.args.get('s4'))
        s5 = float(request.args.get('s5'))
        s6 = float(request.args.get('s6'))
        pred = predict(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
        return json.dumps([{'prediction': pred[0]}])
    else:
        return abort(400, "some parameters are missing, you need: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6")



def predict(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    estimator = estimatorMap.get("estimator")
    if(estimator==None):
        with open('diabetes_model', 'rb') as f:
            estimator = pickle.load(f)
            estimatorMap["estimator"]=estimator
    data = {'age': [age], 'sex': [sex], 'bmi': [bmi], 'bp': [bp], 
            's1': [s1], 's2': [s2], 's3': [s3], 's4': [s4], 's5': [s5], 's6': [s6]}
    return estimator.predict(pd.DataFrame.from_dict(data))

