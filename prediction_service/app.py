import os
import pickle

import requests
from flask import Flask
from flask import request
from flask import jsonify

from pymongo import MongoClient


MODEL_FILE = os.getenv('MODEL_FILE', 'lr.bin')

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('price')
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")


@app.route('/predict', methods=['POST'])
def predict():
    house = house.get_json()

    X = dv.transform([house])
    y_pred = model.predict(X)
    
    result = {
        'house price': float(y_pred),
    }
    
    save_to_db(house, float(y_pred))
    send_to_evidently_service(house, float(y_pred))
    return jsonify(result)


def save_to_db(house, prediction):
    rec = house.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(house, prediction):
    rec = house.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/houses", json=[rec])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)