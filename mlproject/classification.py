import pandas as pd

"""DATA READER: uncomment to see data read from file"""
# load the training data
data = pd.read_csv("./train.csv")
#print data['id'][6]
#print data['comment_text'][6]

# create one vector for the six categories of toxicity
data['toxicity_vec'] = [[a, b, c, d, e, f] for a, b, c, d, e, f in zip(data['toxic'], data['severe_toxic'], data['obscene'], data['threat'], data['insult'], data['identity_hate'])]
#print data['toxicity_vec'][6]
