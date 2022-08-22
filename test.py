import requests
import predict
house={
    "host_identity_verified":'unconfirmed',
    "neighbourhood group":'Bronx',
    "instant_bookable":'False', 
    "cancellation_policy":'moderate', 
    "room type":'Shared room',
    "Construction year":2007,
    "service fee":140, 
    "minimum nights":2,
    "number of reviews":2
}


features=predict.prepare_features(house)
pred=predict.predict(features)
print(pred)
url = 'http://localhost:9696/predict'
response = requests.post(url, json=house)
print(response.json())
