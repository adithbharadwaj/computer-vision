
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage 
from lr_utils import load_dataset # loading the dataset.
from scipy import misc # for loading the image

from PIL import Image as im
from sklearn.metrics import accuracy_score

def sigmoid(z):
    
    s = 1.0 / (1 + np.exp(-z))
    return s

def relu(x):

    return np.maximum(0,x) 


def init_params(layers):

	np.random.seed(1234)

	w1 = np.random.randn(layers[1], layers[0])* np.sqrt(2/layers[0])
	b1 = np.random.randn(layers[1], 1)

	w2 = np.random.randn(layers[2], layers[1]) * np.sqrt(2/layers[1])
	b2 = np.random.randn(layers[2], 1)

	w3 = np.random.randn(layers[3], layers[2]) * np.sqrt(2/layers[2])
	b3 = np.random.randn(layers[3], 1) 

	parameters = {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}

	return parameters

def optimize(parameters, X, Y, num_iterations, learning_rate = 0.001):

	costs = []

	for i in range(num_iterations):

		m = X.shape[1]
	    
		W1 = parameters["W1"]
		b1 = parameters["b1"]
	    
		W2 = parameters["W2"]
		b2 = parameters["b2"]
	    
		W3 = parameters["W3"]
		b3 = parameters["b3"]

	    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
		Z1 = np.dot(W1, X) + b1
		A1 = relu(Z1)

		Z2 = np.dot(W2, A1) + b2
		A2 = relu(Z2)
	    
		Z3 = np.dot(W3, A2) + b3
		A3 = sigmoid(Z3)



		m = X.shape[1]
	    
		dZ3 = A3 - Y
		dW3 = 1. / m * np.dot(dZ3, A2.T)
		db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
	    
		dA2 = np.dot(W3.T, dZ3)
		dZ2 = np.multiply(dA2, np.int64(A2 > 0))
		dW2 = 1. / m * np.dot(dZ2, A1.T)
		db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
	    
		dA1 = np.dot(W2.T, dZ2)
		dZ1 = np.multiply(dA1, np.int64(A1 > 0))
		dW1 = 1. / m * np.dot(dZ1, X.T)
		db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)



		parameters["W3"] = parameters["W3"] - learning_rate* dW3
		parameters["b3"] = parameters["b3"] - learning_rate * db3

		parameters["W2"] = parameters["W2"] - learning_rate*dW2
		parameters["b2"] = parameters["b2"] - learning_rate*db2

		parameters["W1"] = parameters["W1"] - learning_rate*dW1
		parameters["b1"] = parameters["b1"] - learning_rate*db1

		cost = -np.sum((Y * np.log(A3) + (1 - Y)* np.log(1 - A3)))/m

		if(i % 100 == 0):
			costs.append(cost)
			print("cost after iteration " , i , "is: " , cost)

	return parameters, cost    

def predict(parameters, X):

		W1 = parameters["W1"]
		b1 = parameters["b1"]
	    
		W2 = parameters["W2"]
		b2 = parameters["b2"]
	    
		W3 = parameters["W3"]
		b3 = parameters["b3"]

	    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
		Z1 = np.dot(W1, X) + b1
		A1 = relu(Z1)

		Z2 = np.dot(W2, A1) + b2
		A2 = relu(Z2)
	    
		Z3 = np.dot(W3, A2) + b3
		A3 = sigmoid(Z3)

		return A3


if __name__ == '__main__':
	
	train_set_x_orig, Y_train, test_set_x_orig, Y_test, classes = load_dataset()	

	index = 25
	
	'''
	uncomment this to visualize a particular example in the dataset.
	plt.imshow(x_train_orig[index])
	print ("y = " + str(Y_train[:,index]) + ", it's a '" + classes[np.squeeze(Y_train[:,index])].decode("utf-8") +  "' picture.")
	plt.show()
	'''

	# reshaping the data. same as reshaping (a, b, c, d) to (b*c*d, a)

	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

	# normalizing the values because the max value of a pixel is 255.  
	x_train = train_set_x_flatten/255.
	x_test = test_set_x = test_set_x_flatten/255.

	num_iterations = 8000
	learning_rate = 0.006

	layers = [x_train.shape[0], 20, 5, 1]

	params = init_params(layers)

	params, cost = optimize(params, x_train, Y_train, num_iterations, learning_rate)

	'''cat = im.open('cat2.jpg')
	cat.show()
	cat = cat.resize((64, 64))
	'''

	cat = misc.imread('cat2.jpg')
	plt.imshow(cat)
	plt.show()

	cat = misc.imresize(cat, (64, 64))
	cat = cat.reshape(cat.shape[0]*cat.shape[1]*cat.shape[2], 1)

	pred = predict(params, cat)
	print(np.round(pred))

	'''
	arya = misc.imread('arya.jpg')
	arya = misc.imresize(arya, (64, 64))
	arya = arya.reshape(arya.shape[0]*arya.shape[1]*arya.shape[2], 1)

	a = im.open('arya.jpg').show()

	pred = predict(params, arya)
	print(np.round(pred))
    '''
	test = misc.imread('cat.jpg')
	plt.imshow(test)
	plt.show()

	test = test.reshape(test.shape[0]*test.shape[1]*test.shape[2], 1)
	pred = predict(params, test)
	print(np.round(pred))

	test_pred = predict(params, x_test)
	test_pred = np.round(test_pred)

	score = 0
	
	for i in range(len(test_pred[0])):
		if(test_pred[0][i] == Y_test[0][i]):
			score += 1

	print("2 hidden layer network ", layers)		

	print("accuracy: % ", float(score/len(test_pred[0])*100))		

	print("sklearn accuracy score: ", accuracy_score(Y_test[0], test_pred[0]))