# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:35:41 2018
Cite: 
@author: chloechen
"""

import torch
from torch.autograd import Variable
import pandas as pd

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 10, 4, 2, 3

#load
datatrain = pd.read_csv('iris_train.csv')

#change string value to numeric
datatrain.set_value(datatrain['species']=='Iris-setosa',['species'],0)
datatrain.set_value(datatrain['species']=='Iris-versicolor',['species'],1)
datatrain.set_value(datatrain['species']=='Iris-virginica',['species'],2)
datatrain = datatrain.apply(pd.to_numeric)

#change dataframe to array
datatrain_array = datatrain.as_matrix()
#print datatrain_array[0]

#split x and y (feature and target)
xtrain = datatrain_array[:,:4]
ytrain = datatrain_array[:,4]

ytrain1hot = []
for i in ytrain:
    if i == 0:
        ytrain1hot.append((0,0,0))
    elif i == 1:
        ytrain1hot.append((0,0,1))
    else:
        ytrain1hot.append((0,1,1))
#print xtrain[0]
#print ytrain1hot[0]

x = Variable(torch.Tensor(xtrain).float())
y = Variable(torch.Tensor(ytrain1hot).float())

# Use the nn package to define our model as a sequence of layers
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.LogSoftmax()
)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(50):
    # Forward pass: pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)
    #print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradients for all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data
        
#load
datatest = pd.read_csv('iris_test.csv')

#change string value to numeric
datatest.set_value(datatest['species']=='Iris-setosa',['species'],0)
datatest.set_value(datatest['species']=='Iris-versicolor',['species'],1)
datatest.set_value(datatest['species']=='Iris-virginica',['species'],2)
datatest = datatest.apply(pd.to_numeric)

#change dataframe to array
datatest_array = datatest.as_matrix()
#print datatest_array[0]

#split x and y (feature and target)
x_test = datatest_array[:,:4]
y_test = datatest_array[:,4]
#print xtrain[0]
#print ytrain[0]

ytest1hot = []
for i in y_test:
    if i == 0:
        ytest1hot.append((0,0,0))
    elif i == 1:
        ytest1hot.append((0,0,1))
    else:
        ytest1hot.append((0,1,1))
print y_test
print ytest1hot

xtest = Variable(torch.Tensor(x_test).float())
ytest = Variable(torch.Tensor(y_test).float())
ypred = model(xtest)

_, predicted = torch.max(ypred.data, 1)

print predicted

#get accuracy
for x in ypred:
    predset = list(x.data)
    print sorted(predset)[2]
    
#print(ytest)
print('Accuracy of the network %d %%' % (100 * torch.sum(y_test==predicted) / 30))
