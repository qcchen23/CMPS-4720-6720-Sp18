import pandas as pd
import numpy as np
import nltk
import re
import logging
import HTMLParser
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
import gensim
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

stop = set(stopwords.words('english'))
html_parser = HTMLParser.HTMLParser()
tknzr = TweetTokenizer()

stop.add('...')
stop.add('..')
stop.add('::')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""DATA READER"""
# load the training data
data = pd.read_csv("train.csv")
#print data['id'][6]
#print data['comment_text'][6]

# create one vector for the six categories of toxicity
data['toxicity_vec'] = [[a, b, c, d, e, f] for a, b, c, d, e, f in zip(data['toxic'], data['severe_toxic'], data['obscene'], data['threat'], data['insult'], data['identity_hate'])]
#print data['toxicity_vec'][6]



def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    # calculate the mean vector for each tweet
    doc = [word for word in doc if word in word2vec_model.vocab]
    if len(doc) > 0:
        return np.mean(word2vec_model[doc], axis=0)
    else:
        #print np.zeros(1)
        return np.zeros(300)


# create clean text data
t = data['comment_text']

sent_toks=[]
clean_tweets = []

for tweet in t:
    ctweet = ''.join(i.lower() for i in tweet if ord(i) < 128 and i != '#')
    ctweet = ctweet.decode('utf8').encode('ascii','strict')
    ctweet = html_parser.unescape(ctweet)
    tok = tknzr.tokenize(ctweet)
    # tokenize and take out @user tags
    toks =[]
    for t in tok:
        if t[0] != '@' and t not in stop and len(t)>1:
            toks.append(t)
    sent_toks.append(toks)
    # collect all clean tweets
    clean_tweets.append(ctweet)

# add 'clean tweet' and 'tokens' as new columns into dataframe
data['clean tweet'] = clean_tweets
data['tokens'] = sent_toks

logger.info("Data preprocessing complete.")

#t = t.apply(lambda x: [item.str.strip('\n') for item in x if item != "" and item not in stop])
#print data['tokens']


# load model directly from file, after previous code has been run once
model = gensim.models.KeyedVectors.load("wordvectors.txt")
logger.info("Model read from file.")

# generate one mean vector for each tweet, based on model
mean_vecs = []
for s in range(len(data['tokens'])):
    mean = document_vector(model,data['tokens'][s])
    mean_vecs.append(mean)

# print mean vecs to file for faster read
with open('mean_vecs.txt','w') as f:
    f.writelines('\t'.join(str(j) for j in i) +'\n' for i in mean_vecs)

# read mean vecs
with open('mean_vecs.txt') as t:
    mean_vecs = [map(float, line.split()) for line in t]

# PREDICTION
#
#
logger.info("Loading classifier.")
#clf = svm.SVC()
clf = svm.LinearSVC() # for one-vs-all/one-of
#clf = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(mean_vecs, data['toxicity_vec'], test_size=0.1, random_state=42)


#fit our classifier
clf.fit(X_train,y_train)

#predict on our test samples
pred = clf.predict(X_test)

print classification_report(y_test, pred)


