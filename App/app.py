import pickle
import pandas as pd
import numpy as np
from model import LR_Model
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder ="static")
vectorizer = CountVectorizer()
# pull skeleton from index.html
@app.route('/', methods=['GET'])
def main_page():
    return render_template("main_page.html")

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    text = str(request.form['text'])
    text_ = loaded_vectorizer.transform([text])
    predict = nb.predict_input(text_)
    if predict[0] == 'NEGATIVE':
        print(predict[1])
        return render_template("negative.html", value=round(predict[1], 3))
    else:  
        print(predict[1])
        return render_template("positive.html",value=round(predict[1],3))
   


if __name__ == '__main__':
    # Load files.
    model = pd.read_pickle('model.pkl')
    with open('model.pkl', 'rb') as f:
        nb = pickle.load(f)
    loaded_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  
 
    app.run(host='0.0.0.0', port=8080, debug=True)