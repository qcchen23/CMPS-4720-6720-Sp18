import numpy as np
import csv

# load the training data
cases, x_train, y_train = [],[],[]
with open('SPECT.train', 'rb') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        case = ', '.join(row)
        cases.append(case)
        
for c in cases:
    x_tn = []
    y_train.append(int(c[0]))
    for i in c[3:]:
        if i != ',' and i != ' ':
            x_tn.append(int(i))
    x_train.append(x_tn)
x_train = np.array(x_train)
y_train = np.array(y_train)

# load the test data
newcases, x_test, y_test = [],[],[]
with open('SPECT.test', 'rb') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        newcase = ', '.join(row)
        newcases.append(newcase)
for c in newcases:
    x_t = []
    y_test.append(int(c[0]))
    for i in c[3:]:
        if i != ',' and i != ' ':
            x_t.append(int(i))
    x_test.append(x_t)
x_test = np.array(x_test)
y_test = np.array(y_test)

# make a perceptron class with fit() and predict() methods
class Perceptron(object):
    def __init__(self, r=0.01, n=10):
        self.learningRate = r
        self.itera = n

    def fit(self, X, y):
        # make an np array of zeros
        self.weights = np.zeros(1 + X.shape[1])
        self.err = []
        for i in range(self.itera):
            errors = 0
            for k, target in zip(X, y):
                # update the weight in each iteration
                update = self.learningRate * (target - self.predict(k))
                self.weights[1:] += update * k
                self.weights[0] += update
                errors += int(update != 0.0)
            self.err.append(errors)
        return self

    def net_input(self, X):
        # get the dot product 
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        
p = Perceptron(0.01,10)
p.fit(x_train, y_train)
y_pred = p.predict(x_test)
print "preditions:"
print y_pred
print "actualy labels:"
print y_test
print "evaluation:"

counta,countb = 0, 0
for i in range(len(y_test)):
    if y_pred[i] == 0 and y_test[i] == 0:
        counta += 1
    elif y_pred[i] == 1 and y_test[i] == 1:
        countb += 1
print "accuracy is: " + str((counta + countb)/ float(len(y_test)) * 100) +"%"
