# Justine Ayroor 
# UCID: JA573
# Assignment 5 

import numpy as np
import sys
import pandas as pd
from scipy import signal
import math

#################
### Read data ###

traindir = sys.argv[1]
df = pd.read_csv(traindir+'/data.csv')
names = df['Name'].values
labels = df['Label'].values

# trainData = np.empty((len(labels),3,3))
trainData = np.empty((len(labels),3,3),dtype=np.float)
for i in range(0,len(labels)):
    image_matrix = np.loadtxt(traindir+'/'+names[i])
    trainData[i] = image_matrix

testdir = sys.argv[2]
df = pd.read_csv(testdir+'/data.csv')
names = df['Name'].values
testData = np.empty((len(names),3,3),dtype=np.float)
for i in range(0,len(labels)):
    image_matrix = np.loadtxt(traindir+'/'+names[i])
    testData[i] = image_matrix

sigmoid = lambda x: 1/(1+np.exp(-x))

# ##############################
# ### Initialize all weights ###

# c = np.random.rand(2,2)
c = np.ones((2,2),dtype=np.float)
# c = np.ones((2,2))
# c = np.ones((2,2))

epochs = 10000
eta = 0.1
obj = 0
prevObj = np.inf

# ###########################
# ### Calculate objective ###

for i in range(0,len(labels)):
    hidden_layer = signal.convolve2d(trainData[i], c, mode="valid")
    print(hidden_layer)

    for j in range(0,2,1):
        for k in range(0,2,1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
        
    output_layer = (hidden_layer[0][0]+hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
    print("output_layer",output_layer)
    obj+=(output_layer-labels[i])**2

# ###############################
# ### Begin gradient descent ####
# stop=0.0000001
stop = 0.01
# stop = 0
# while(prevObj - obj > stop and i << epochs):
while (prevObj - obj >= stop):
# while epochs != 0:
    
    # update previous obj
    prevObj = obj

    # print("c=",c)
    dellc1 = 0
    dellc2 = 0
    dellc3 = 0
    dellc4 = 0

    for i in range(0,len(labels),1):
        
        # Convolution
        hidden_layer = signal.convolve2d(trainData[i],c,mode="valid")
        for j in range(0,2,1):
            for k in range(0,2,1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])

        # Calculate gradient descent for c1
        sqrtf = ((hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1])/4) - labels[i]
        dz1dc1 = hidden_layer[0][0]*(1-hidden_layer[0][0])*trainData[i][0][0] # p1
        dz2dc1 = hidden_layer[0][1]*(1-hidden_layer[0][1])*trainData[i][0][1] # p2
        dz3dc1 = hidden_layer[1][0]*(1-hidden_layer[1][0])*trainData[i][1][0] # p4
        dz4dc1 = hidden_layer[1][1]*(1-hidden_layer[1][1])*trainData[i][1][1] # p5
        dellc1 += (sqrtf *(dz1dc1 + dz2dc1 + dz3dc1 + dz4dc1))/2

        # Calculate gradient descent for c2
        dz1dc2 = hidden_layer[0][0]*(1-hidden_layer[0][0])*trainData[i][0][1] # p2
        dz2dc2 = hidden_layer[0][1]*(1-hidden_layer[0][1])*trainData[i][0][2] # p3
        dz3dc2 = hidden_layer[1][0]*(1-hidden_layer[1][0])*trainData[i][1][1] # p5
        dz4dc2 = hidden_layer[1][1]*(1-hidden_layer[1][1])*trainData[i][1][2] # p6
        dellc2 += (sqrtf * ((dz1dc2 + dz2dc2 + dz3dc2 + dz4dc2)))/2

        # Calculate gradient descent for c3
        dz1dc3 = hidden_layer[0][0]*(1-hidden_layer[0][0])*trainData[i][1][0] # p4
        dz2dc3 = hidden_layer[0][1]*(1-hidden_layer[0][1])*trainData[i][1][1] # p5
        dz3dc3 = hidden_layer[1][0]*(1-hidden_layer[1][0])*trainData[i][2][0] # p7
        dz4dc3 = hidden_layer[1][1]*(1-hidden_layer[1][1])*trainData[i][2][1] # p8
        dellc3 += (sqrtf * ((dz1dc3 + dz2dc3 + dz3dc3 + dz4dc3)))

        # Calculate gradient descent for c4
        dz1dc4 = hidden_layer[0][0]*(1-hidden_layer[0][0])*trainData[i][1][1] # p5
        dz2dc4 = hidden_layer[0][1]*(1-hidden_layer[0][1])*trainData[i][1][2] # p6
        dz3dc4 = hidden_layer[1][0]*(1-hidden_layer[1][0])*trainData[i][2][1] # p8
        dz4dc4 = hidden_layer[1][1]*(1-hidden_layer[1][1])*trainData[i][2][2] # p9
        dellc4 +=  (sqrtf *((dz1dc4 + dz2dc4 + dz3dc4 + dz4dc4)))/2

    # update c1,c2,c3,c4
    c[0][0]-=eta*dellc1
    c[0][1]-=eta*dellc2
    c[1][0]-=eta*dellc3
    c[1][1]-=eta*dellc4

    # Recalculate obj
    obj=0
    for i in range(0,len(labels),1):
        hidden_layer = signal.convolve2d(trainData[i], c, mode="valid")
        
        for j in range(0,2,1):
            for k in range(0,2,1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
        
        output_layer = (hidden_layer[0][0]+hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
        obj+=(output_layer-labels[i])**2

    print("obj = ",obj)

    # epochs -=1
print("c:",c)

# ### Do final predictions ###
# #TODO: get the final predictions

predictions = []
for i in range(0,len(labels),1):
    hidden_layer = signal.convolve2d(testData[i], c, mode="valid")
    for j in range(0,2,1):
        for k in range(0,2,1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    if(((hidden_layer[0][0]+hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4) < 0.5):
        predictions.append(-1)
    else:
        predictions.append(1)
print("Predictions:",predictions)