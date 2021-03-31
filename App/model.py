import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression 

def Train_Test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    return X_train, X_test, y_train, y_test

class LR_Model():
    '''
    Creates a simple Naive Bayes Model
    '''
    def __init__(self):
       # self.nb = MultinomialNB()
        self.LR = LogisticRegression()
        
        
    def fit(self, X_train, y_train):
       # self.nb.fit(X_train, y_train)
        self.LR.fit(X_train, y_train)
    def predict(self, X_test):
        y_predict = self.LR.predict(X_test)
        return y_predict
    def predict_input(self,text_):
        score = self.LR.predict_proba(text_)
        for prob in score:
            max_index = np.argmax(prob)
            probability = prob[max_index]
        
            if max_index == 1:
                return ['POSITIVE', probability]
            else:
                return ['NEGATIVE', probability]
       

if __name__=='__main__':
    samp = pd.read_csv('samp.csv')
    samp = samp.drop('Unnamed: 0', axis = 1)
    samp = samp.dropna()
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    X = vectorizer.fit_transform(samp['text'])
    y = samp['sentiment']
    X_train, X_test, y_train, y_test = Train_Test(X,y)
    LR = LR_Model()
    LR.fit(X_train, y_train)
    
   # text_ = vectorizer.transform([text])
    vec_file = 'vectorizer.pkl'
    pickle.dump(vectorizer, open(vec_file, 'wb'))
    y_predict = LR.predict(X_test)
    #print(y_predict[:10])
    #print(nb.predict_input(text_))
    with open('model.pkl', 'wb') as f:
        #write the model to a file
        pickle.dump(LR,f)
