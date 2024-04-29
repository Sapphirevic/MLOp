import predict

price = {"BedroomAbvGr": 4, "GarageCars": 2, "SalePrice": 250000}

features = predict.prepare_features(price)
pred = predict.predict(features)
print(pred)

# with open("models/lasso.bin", "rb") as f_in:
#     (dv, model) = pickle.load(f_in)


# def predict(features):
#     X = dv.transform(features)
#     preds = model.predict(X)
#     return preds
