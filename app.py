from flask import Flask
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)



with open("rdcmodel.pkl","rb") as f:
    model = pickle.load(f)

with open("leencoder.pkl","rb") as f:
    le = pickle.load(f)


@app.route("/")
def home():
    return "Model is Running"

@app.route("/predict",methods=["GET"])
def predict():
    pred = model.predict([[5.1,3.5,1.4,0.2]])
    flower_name = le.inverse_transform(pred)
    return f"Prediction: {flower_name}"

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)