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

rows = train.shape[0]
cols = train.shape[1]

#hidden_nodes = int(sys.argv[3])

hidden_nodes = 3

##############################
### Initialize all weights ###

w = np.random.rand(hidden_nodes)
#w = np.ones(hidden_nodes)
# print("w=",w)

#check this command
#W = np.zeros((hidden_nodes, cols), dtype=float)
#W = np.ones((hidden_nodes, cols), dtype=float)
W = np.random.rand(hidden_nodes, cols)
# print("W=",W)

epochs = 10000
eta = .01
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
# print("hidden_layer=",hidden_layer)
# print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
# print("hidden_layer=",hidden_layer)
# print("hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
# print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
# print("obj=",obj)

#obj = np.sum(np.square(np.matmul(train, np.transpose(w)) - trainlabels))

# print("Obj=",obj)

###############################
### Begin gradient descent ####
# stop=0.0000001
stop = 0
while(prevobj - obj > stop and i < epochs):
#while(prevobj - obj > 0):

	#Update previous objective
	prevobj = obj

	#Calculate gradient update for final layer (w)
	#dellw is the same dimension as w

#	print(hidden_layer[0,:].shape, w.shape)

	dellw = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*hidden_layer[0,:]
	for j in range(1, rows):
		dellw += (np.dot(hidden_layer[j,:],np.transpose(w))-trainlabels[j])*hidden_layer[j,:]

	#Update w
	w = w - eta*dellw
#	print("w=",w)
#	print("dellf=",dellf)
	
	#Calculate gradient update for hidden layer weights (W)
	#dellW has to be of same dimension as W

	#Let's first calculate dells. After that we do dellu and dellv.
	#Here s, u, and v are the three hidden nodes
	#dells = df/dz1 * (dz1/ds1, dz1,ds2)
	dells = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[0] * (hidden_layer[0,0])*(1-hidden_layer[0,0])*train[0]
	for j in range(1, rows):
		dells += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[0] * (hidden_layer[j,0])*(1-hidden_layer[j,0])*train[j]

	#TODO: dellu = ?
	dellu = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[1] * (hidden_layer[0,1])*(1-hidden_layer[0,1])*train[0]
	for j in range(1, rows):
		dellu += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[1] * (hidden_layer[j,1])*(1-hidden_layer[j,1])*train[j]

	#TODO: dellv = ?
	dellv = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[2] * (hidden_layer[0,2])*(1-hidden_layer[0,2])*train[0]
	for j in range(1, rows):
		dellv += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[2] * (hidden_layer[j,2])*(1-hidden_layer[j,2])*train[j]

	# print("dells shape",dells)
	# print("dellu shape",dellu)
	# print("dellv shape",dellv)
	# print("W shape",W.shape)

	#TODO: Put dells, dellu, and dellv as rows of dellW
	dellW = np.empty((0,cols),float)
	
	dellW = np.vstack((dellW,dells,dellu,dellv))
		
	#Update W
	W = W - eta*dellW

	#Recalculate objective
	hidden_layer = np.matmul(train, np.transpose(W))
	# print("hidden_layer=",hidden_layer)

	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
	# print("hidden_layer=",hidden_layer)

	output_layer = np.matmul(hidden_layer, np.transpose(w))
	# print("output_layer=",output_layer)

	obj = np.sum(np.square(output_layer - trainlabels))
	# print("obj=",obj)
	
	i = i + 1
	print("Objective=",obj)


### Do final predictions ###
#TODO: get the final predictions
print(w)
f_layer = np.matmul(train, np.transpose(W))
predictions = np.sign(np.matmul(sigmoid(f_layer), np.transpose(w)))
print(predictions)
