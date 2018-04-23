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
from sklearn.model_selection import cross_val_score

stop = set(stopwords.words('english'))
html_parser = HTMLParser.HTMLParser()
tknzr = TweetTokenizer()

stop.add('...')
stop.add('..')
stop.add('::')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""DATA READER"""

# load data

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('train.csv').fillna(' ')
test = pd.read_csv('test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

# create one vector for the six categories of toxicity
#data['toxicity_vec'] = [[a, b, c, d, e, f] for a, b, c, d, e, f in zip(data['toxic'], data['severe_toxic'], data['obscene'], data['threat'], data['insult'], data['identity_hate'])]

logger.info("Data read.")

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    # calculate the mean vector for each tweet
    doc = [word for word in doc if word in word2vec_model.vocab]
    if len(doc) > 0:
        return np.mean(word2vec_model[doc], axis=0)
    else:
        #print np.zeros(1)
        return np.zeros(300)

sent_toks_train, sent_toks_test=[],[]
clean_tweets = []

for tweet in train_text:
    ctweet = ''.join(i.lower() for i in tweet if ord(i) < 128 and i != '#')
    ctweet = ctweet.decode('utf8').encode('ascii','strict')
    ctweet = html_parser.unescape(ctweet)
    tok = tknzr.tokenize(ctweet)
    # tokenize and take out @user tags
    toks =[]
    for t in tok:
        if t[0] != '@' and t not in stop and len(t)>1:
            toks.append(t)
    sent_toks_train.append(toks)
    # collect all clean tweets
    clean_tweets.append(ctweet)

for tweet in test_text:
    ctweet = ''.join(i.lower() for i in tweet if ord(i) < 128 and i != '#')
    ctweet = ctweet.decode('utf8').encode('ascii','strict')
    ctweet = html_parser.unescape(ctweet)
    tok = tknzr.tokenize(ctweet)
    # tokenize and take out @user tags
    toks =[]
    for t in tok:
        if t[0] != '@' and t not in stop and len(t)>1:
            toks.append(t)
    sent_toks_test.append(toks)
    # collect all clean tweets
    clean_tweets.append(ctweet)

logger.info("Data preprocessing complete.")


"""
# load model directly from file, after previous code has been run once
model = gensim.models.KeyedVectors.load("wordvectors.txt")
logger.info("Model read from file.")

# generate one mean vector for each tweet, based on model
mean_vecs_train = []
for s in range(len(sent_toks_train)):
    mean = document_vector(model,sent_toks_train[s])
    mean_vecs_train.append(mean)

mean_vecs_test = []
for s in range(len(sent_toks_test)):
    mean = document_vector(model,sent_toks_test[s])
    mean_vecs_test.append(mean)

# print mean vecs to file for faster read
with open('mean_vecs_train.txt','w') as f:
    f.writelines('\t'.join(str(j) for j in i) +'\n' for i in mean_vecs_train)

# print mean vecs to file for faster read
with open('mean_vecs_test.txt','w') as f:
    f.writelines('\t'.join(str(j) for j in i) +'\n' for i in mean_vecs_test)
"""

# read mean vecs
with open('mean_vecs_train.txt') as t:
    mean_vecs_train = [map(float, line.split()) for line in t]

# read mean vecs
with open('mean_vecs_test.txt') as s:
    mean_vecs_test = [map(float, line.split()) for line in s]


scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    print "svm"
    classifier = svm.SVC()

    cv_score = np.mean(cross_val_score(classifier, mean_vecs_train, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    print "fitting"
    classifier.fit(mean_vecs_train, train_target)
    submission[class_name] = classifier.decision_function(mean_vecs_test)

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)

