import pickle

from flask import Flask, request, jsonify

with open('lr.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(house):
    features=house.copy()
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    house = house.get_json()

    features = prepare_features(house)
    pred = predict(features)

    result = {
        'house price': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)