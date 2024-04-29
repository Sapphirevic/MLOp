import pickle
from flask import Flask, request, jsonify

# if not os.path.exists("models"):
#     os.makedirs("models")

# original code to save the file
with open("models/lasso.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(price):
    features = {}
    features["BED_GAR"] = "%s_%s" % (price["BedroomAbvGr"], price["GarageCars"])
    features["SalePrice"] = price["SalePrice"]
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds


app = Flask("Pricing-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    price = request.get_json()

    features = prepare_features(price)
    pred = predict(features)

    result = {"SalePrice": pred}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
