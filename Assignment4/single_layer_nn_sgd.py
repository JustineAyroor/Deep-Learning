# Justine Ayroor
# Assignment 4
# 3 command line arguments 
# Please execute as follows
# python single_layer_nn_sgd.py ion.train.0.txt ion.test.0.txt 25
# 1st argv : train dataset
# 2nd argv : test dataset
# 3rd argv : batch-size for sgt
import numpy as np
import sys

#################
### Read data ###

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

# print("train=",train)
# print("train shape=",train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

rows = train.shape[0]
cols = train.shape[1]

miniBatchlen = int(sys.argv[3])
#hidden_nodes = int(sys.argv[3])

hidden_nodes = 3

##############################
### Initialize all weights ###

w = np.random.rand(hidden_nodes)
W = np.random.rand(hidden_nodes, cols)
epochs = 10000
# Set According to dataset
eta = .001
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])

output_layer = np.matmul(hidden_layer, np.transpose(w))

obj = np.sum(np.square(output_layer - trainlabels))

###############################
### Begin gradient descent ####
stop=0.00001
# stop = 0
rowindex = np.array([i for i in range(rows)])
while(prevobj - obj > stop and i < epochs):
        #Update previous objective
        prevobj = obj

        for k in range(0, rows, 1):
        #Calculate gradient update for final layer (w)
        #dellw is the same dimension as w
            np.random.shuffle(rowindex)
            index = rowindex[0]
            dellw = (np.dot(hidden_layer[index,:],np.transpose(w))-trainlabels[index])*hidden_layer[index,:]
            for j in range(1, miniBatchlen):
                index = rowindex[j]
                dellw += (np.dot(hidden_layer[index,:],np.transpose(w))-trainlabels[index])*hidden_layer[index,:]

            #Update w
            w = w - eta*dellw
            #Calculate gradient update for hidden layer weights (W)
            #dellW has to be of same dimension as W

            #Let's first calculate dells. After that we do dellu and dellv.
            #Here s, u, and v are the three hidden nodes
            #dells = df/dz1 * (dz1/ds1, dz1,ds2)
            index = rowindex[0]
            dells = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[0] * (hidden_layer[index,0])*(1-hidden_layer[index,0])*train[index]
            for j in range(1, miniBatchlen):
                index = rowindex[j]
                dells += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[0] * (hidden_layer[index,0])*(1-hidden_layer[index,0])*train[index]

            #TODO: dellu = ?
            index = rowindex[0]
            dellu = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[1] * (hidden_layer[index,1])*(1-hidden_layer[index,1])*train[index]
            for j in range(1, miniBatchlen):
                index = rowindex[j]
                dellu += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[1] * (hidden_layer[index,1])*(1-hidden_layer[index,1])*train[index]

            #TODO: dellv = ?
            index = rowindex[0]
            dellv = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[2] * (hidden_layer[index,2])*(1-hidden_layer[index,2])*train[index]
            for j in range(1, miniBatchlen):
                index = rowindex[j]
                dellv += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[2] * (hidden_layer[index,2])*(1-hidden_layer[index,2])*train[index]

            #TODO: Put dells, dellu, and dellv as rows of dellW
            dellW = np.empty((0,cols),float)

            dellW = np.vstack((dellW,dells,dellu,dellv))

            #Update W
            W = W - eta*dellW

        #Recalculate objective
        hidden_layer = np.matmul(train, np.transpose(W))
        hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
        output_layer = np.matmul(hidden_layer, np.transpose(w))
        obj = np.sum(np.square(output_layer - trainlabels))
        i = i + 1
        print("Objective=",obj)


### Do final predictions ###
#TODO: get the final predictions
print(w)
f_layer = np.matmul(test, np.transpose(W))
predictions = np.sign(np.matmul(sigmoid(f_layer), np.transpose(w)))
error = 1 - np.mean(predictions == testlabels)
print("Error=",error)
# print(predictions)
