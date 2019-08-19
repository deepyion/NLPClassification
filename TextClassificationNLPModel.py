
# coding: utf-8

# In[1]:


#importing required packages
import pandas as pd
import numpy as np
import json 
import sys
import os
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import re
from sklearn import metrics
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from spacy import displacy
from spacy.util import minibatch, compounding
import seaborn as sns


# In[2]:


#read data to python notebook from local file system
with open("trainingSet.json") as jf:
    data = json.load(jf)
    print(data[:5])


# In[3]:


#Convert to dataframe
df1 = pd.DataFrame(data)
df1.columns = ["text", "category"]


# In[4]:


df1.head()


# In[5]:


#Check the dimensions of data: number of rows(497) and columns(2)
df1.shape


# In[6]:


#Any missing values in the dataset
df1.isna().sum()


# There are no missing values

# In[7]:


#In the taget how many unique classes are present
df1['category'].unique()


# In[8]:



sns.countplot(df1.category)
plt.xticks(rotation=90)


# In[9]:


fig = plt.figure(figsize=(8,4))
sns.barplot(x = df1['category'].unique(), y=df1['category'].value_counts())
plt.xticks(rotation=90)
plt.show()


# In[10]:


nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation


# ## Text Preprocessing

# In[11]:


def cleanup_text(docs, logging=False):
    ''' This function takes dict/list return a dict'''
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000==0 and logging:
            print("Processed %d out of %d documents." %(counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

INFO_text = [text for text in df1[df1['category']=='CHARGES']['text']]

INFO_clean = cleanup_text(INFO_text)
INFO_clean = ' '.join(INFO_clean).split()

INFO_counts = Counter(INFO_clean)

INFO_common_words = [word[0] for word in INFO_counts.most_common(20)]
INFO_common_counts = [word[1] for word in INFO_counts.most_common(20)]



# In[12]:


#Explore the data distribution in each class
fig = plt.figure(figsize=(20,20))
sns.barplot(x=INFO_common_words, y=INFO_common_counts)
plt.title('Most Common Words used in the report for Charges related')
plt.xticks(rotation=90)
plt.show()


# # Use spacy to perform the preprocessing

# In[13]:


# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    """ Takes each text performs all the actions like removing stopwords, punctuations..
    and returns meaningful words"""
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
# Custom transformer using spaCy
class predictors(TransformerMixin):
    """returns the cleaned text after transforming anf fitting"""
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


# In[15]:


#Bag of words object(bow_vector) is created
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))


# In[16]:


#Split the dataset into train and test
from sklearn.model_selection import train_test_split

X = df1['text'] # the features we want to analyze
ylabels = df1['category'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)


# # Logistic Regression Model

# In[17]:


# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)


# In[18]:


# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted, average = 'micro'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average = 'micro'))


# # LinearSVC model

# In[19]:


#Packages for building SVC model
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score


# In[32]:


STOPLIST = set(stopwords + list(ENGLISH_STOP_WORDS))


# In[33]:


STOPLIST = set(stopwords + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
class CleanTextTransformer(TransformerMixin):
    """ takes the transform parameters, fit_params to act on the train(X) data """
    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
def get_params(self, deep=True):
        return {}
    
def cleanText(text):
    """ Takes the text as input and returns words in lower case after removing newline and \r delimiters"""
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


# In[34]:


#split the data to train and test
train, test = train_test_split(df1, test_size=0.30, random_state=42)


def printNMostInformative(vectorizer, clf, N):
    """Gives the best performing features|words from the text that is passed """
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
clf = LinearSVC()

pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
# data
X_train = train['text'].tolist()
y_train = train['category'].tolist()

X_test = test['text'].tolist()
y_test = test['category'].tolist()

# train
pipe.fit(X_train, y_train)
# test
preds = pipe.predict(X_test)
print("accuracy:", accuracy_score(y_test, preds))


# In[35]:


#Since precision and recall need to be considered we are taking the 'f1-score' as model performance metric
print(metrics.classification_report(y_test, preds))


# In[36]:


#pickle the trained model and save for future predictions
with open(r'issue_classify.pkl', 'wb') as file:
    pickle.dump(pipe, file)

