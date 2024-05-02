import requests

price = {"BedroomAbvGr": 4, "GarageCars": 2, "SalePrice": 250000}

# features = predict.prepare_features(price)
# pred = predict.predict(features)
# print(pred)
# curl 'http://localhost:80/predict'

url = "http://localhost:8080/predict"
response = requests.post(url, json=price)
print(response.json())
# with open("models/lasso.bin", "rb") as f_in:
#     (dv, model) = pickle.load(f_in)


# def predict(features):
#     X = dv.transform(features)
#     preds = model.predict(X)
#     return preds
