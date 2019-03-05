# curl "https://predictdiabetes.azurewebsites.net/api/predict_diabetes?code=qnjlgedSEAB6Xmx2kIgrMfXWmWzbDQEBZJWYgG2/Dew6bQciIahnaQ==&age=0.038076&sex=0.050680&bmi=0.061696&bp=0.021872&s1=-0.044223&s2=-0.034821&s3=-0.043401&s4=-0.002592&s5=0.019908&s6=-0.017646" 

import logging
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import abort
import json
import azure.functions as func

estimatorMap = {}


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('predict_diabetes started')

    age = req.params.get('age')
    sex = req.params.get('sex')
    bmi = req.params.get('bmi')
    bp = req.params.get('bp')
    s1 = req.params.get('s1')
    s2 = req.params.get('s2')
    s3 = req.params.get('s3')
    s4 = req.params.get('s4')
    s5 = req.params.get('s5')
    s6 = req.params.get('s6')
    if(age and sex and bmi and bp and s1 and s2 and s3 and s4 and s5 and s6):
        pred = predict(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
        return func.HttpResponse("Prediction: " + str(pred[0]))
    else:
        return func.HttpResponse(
            "some parameters are missing or wrong, you need: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6 as numbers",
            status_code=400)


def predict(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    estimator = estimatorMap.get("estimator")
    if(estimator == None):
        with open('diabetes_model', 'rb') as f:
            estimator = pickle.load(f)
            estimatorMap["estimator"] = estimator
    data = {'age': [age], 'sex': [sex], 'bmi': [bmi], 'bp': [bp],
            's1': [s1], 's2': [s2], 's3': [s3], 's4': [s4], 's5': [s5], 's6': [s6]}
    return estimator.predict(pd.DataFrame.from_dict(data))
