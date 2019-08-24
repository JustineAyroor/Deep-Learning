import numpy as np
import sys

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

print(train)
print("train shape=",train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

rows = train.shape[0]
cols = train.shape[1]

w = np.random.rand(cols)
#w = np.ones(cols)
print("w=",w)

epochs = 10000
eta = .001
prevobj = np.inf
i=0

obj = np.sum(np.square(np.matmul(train, np.transpose(w)) - trainlabels))
print("Obj=",obj)

while(prevobj - obj > 0 and i < epochs):
#while(prevobj - obj > 0):

	#Update previous objective
	prevobj = obj

	#Calculate gradient
	dellf = (np.dot(train[0,:],w)-trainlabels[0])*train[0,:]
	for j in range(1, rows):
		dellf += (np.dot(train[j,:],w)-trainlabels[j])*train[j,:]

#	print("dellf=",dellf)
	
	#Update w
	w = w - eta*dellf

	#Recalculate objective
	obj = np.sum(np.square(np.matmul(train, np.transpose(w)) - trainlabels))
	
	i = i + 1
	print("Objective=",obj)
	

predictions = np.sign(np.matmul(train, np.transpose(w)))
print(predictions)

print(w)
