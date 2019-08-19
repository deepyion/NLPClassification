import pickle
from flask import Flask,render_template,url_for,request, abort
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import string
import re
import spacy
#spacy.load('en')
from spacy.lang.en import English
parser = English()

app = Flask(__name__)

@app.route('/')
def home():
    """
    First page of application
    """
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    """ 
    predicts the class for the text provided
    text -> predicted class
    """
    app.logger.info("{} request received from: {}".format(
        request.method, request.remote_addr))
    # Extract Feature With CountVectorizer

    if not request.form['message']:
        app.logger.error("Request has no data or request is not json, aborting")
        abort(400)

    if request.method == 'POST':
        message = request.form['message']
        message = [message]
        my_prediction = clf_pipe.predict(message)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
    class CleanTextTransformer(TransformerMixin):
        def transform(self, X, **transform_params):
            return [cleanText(text) for text in X]
        def fit(self, X, y=None, **fit_params):
            return self
    def get_params(self, deep=True):
        return {}
    
    def cleanText(text):
        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = text.lower()
        return text
    def tokenizeText(sample):
        tokens = parser(sample)
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas
        tokens = [tok for tok in tokens if tok not in STOPLIST]
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
        return tokens
    with open('issue_classify.pkl', 'rb') as model:
        clf_pipe = pickle.load(model)
    print("Imported the model created successfully")
    app.run(debug=True)


