from flask import Flask, render_template, request
import pickle
import numpy as np
# print(np.arange(0,10))
app = Flask(__name__)
model = pickle.load(open("bnglore.pkl","rb"))

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/",methods=['GET','POST'])
def predict():
    features = [int(x) for x in request.form.values()]    
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = str(round(prediction[0],2))
    # print(output)
    
    return render_template("index.html",prediction = output)
    # return render_template("index.html")


if __name__=="__main__":
    app.run(host = '0.0.0.0',debug=True)


