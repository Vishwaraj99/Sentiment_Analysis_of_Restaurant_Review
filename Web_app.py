import numpy as np
from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

app = Flask(__name__)

def preprocess_text(text):
    s_list = stopwords.words('english')
    ps = PorterStemmer()
    # remove special characters or digits
    mystr = re.sub(r'[^A-Za-z\s]','',text)
    mystr = mystr.lower()  #lowercase
    # tokenization
    list1 = mystr.split()
    # remove stopwords
    list2 = [i for i in list1 if i not in s_list]
    # stemming
    list3 = [ps.stem(i) for i in list2]
    final_str = ' '.join(list3)
    return final_str

try:
    f1 = open('cv_rest', 'rb')
    cv = pickle.load(f1)
    f2 = open('model_rest', 'rb')
    model = pickle.load(f2)
    f1.close()
    f2.close()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    cv = None
    model = None


@app.route('/')
def home():
   return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
   '''
   For rendering results on HTML GUI
   '''


   if request.method == 'POST':
       text = request.form['Review']
       if not model_loaded:
           return render_template('index.html', prediction_text='Model not loaded. Please train the model first.')
       processed_text = preprocess_text(text)
       data = [processed_text]
       vectorizer = cv.transform(data).toarray()
       prediction = model.predict(vectorizer)
       prediction=prediction[0]
       print(data)
       #res = model.predict(t)
       if ("not" in text) or ("no" in text) or ("n't" in text):
           prediction= abs(prediction - 1)
       print(prediction)
   if prediction==1:
       return render_template('index.html', prediction_text='The review is Positive')
   else:
       return render_template('index.html', prediction_text='The review is Negative.')





if __name__ == "__main__":
    app.run(debug=True)