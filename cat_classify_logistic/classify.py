import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage 
from lr_utils import load_dataset # loading the dataset.
from scipy import misc # for loading the image

# logistic regression can be considered a single layer neural net.

def sigmoid(z):
    
    s = 1.0 / (1 + np.exp(-z))
    return s

#initializing the weights and biases to zero

def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    b = 0

    return w, b

# forward propagation.

def propagate(w, b, X, Y):

	m = X.shape[1] # number of examples.

	a = sigmoid(np.dot(w.T, X) + b)    

	cost = -np.sum((Y * np.log(a) + (1 - Y)* np.log(1 - a)))/m # cross entropy cost function.

	dw = np.dot(X, (a - Y).T)/m # gradient for weights
	db = np.sum(a - Y)/m # gradient for biases.

	cost = np.squeeze(cost) # removes single dimentional entries from the shape of an array.
    
	grads = {"dw": dw, "db": db} # dict to store the gradients.

	return grads, cost

# gradient descent optimizer.

def optimize(w, b, X, Y, num_iterations, learning_rate):

	costs = []

	for i in range(num_iterations):

		grads, cost = propagate(w, b, X, Y)

		dw = grads["dw"]
		db = grads["db"]

		w = w - learning_rate * dw
		b = b - learning_rate * db

		if(i % 100 == 0):
			costs.append(cost)

	params = {"w": w, "b": b}
	grads = {"dw": dw, "db": db}

	return params, grads, costs			

# function to predict.

def predict(w, b, X):

	# basically forward propagating.

    A = sigmoid(np.dot(w.T,X)+b)
    
    prediction = np.round(A)

    return prediction        

# driver function 

if __name__ == '__main__':
	
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()	

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
	X_train = train_set_x_flatten/255.
	X_test = test_set_x = test_set_x_flatten/255.

	num_iterations = 2000
	learning_rate = 0.005

	w, b = initialize_with_zeros(X_train.shape[0])

	parameters, grads, costs = optimize(w, b, X_train, train_set_y, num_iterations, learning_rate)

	w = parameters["w"]
	b = parameters["b"]
    
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

	print(costs[len(costs) - 1])

	test = misc.imread('cat.jpg')
	plt.imshow(test)
	plt.show()

	test = test.reshape(test.shape[0]*test.shape[1]*test.shape[2], 1)
	pred = predict(w, b, test)
	print(np.round(pred))

'''
	uncomment to visualize the cost function. 
	costs = np.squeeze(cost)
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.show()
'''		
