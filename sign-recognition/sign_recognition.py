import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from cnn_utils import *

def init_prams():

    tf.set_random_seed(1)                              # so that your "random" numbers match ours
        
    W1 = tf.get_variable('W1',[4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2',[2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    return W1, W2


def forward_propagate(X, W1, W2):

	#CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    # CONV2D: stride of 1, padding 'SAME'

    X = tf.cast(X, tf.float32)

    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    # FLATTEN

    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

    return Z3
    

def compute_cost(Z3, Y):
   
    #Computes the cost
    
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)
    cost = tf.reduce_mean(cost)
    
    return cost


def train(x_train, y_train, x_test, y_test, learning_rate = 0.009, num_epochs = 100, minibatch_size = 64):

	tf.set_random_seed(1)                  # to keep results consistent (tensorflow seed) 
	(m, n_H0, n_W0, n_C0) = x_train.shape             
	n_y = y_train.shape[1]      

	seed = 3                      
    
	x = tf.placeholder('float32', [None, n_H0, n_W0, n_C0])
	y = tf.placeholder('float32', [None, n_y])

	costs = []  

	w1, w2 = init_prams()

	z3 = forward_propagate(x_train, w1, w2)

	cost = compute_cost(z3, y_train)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)    

	init = tf.global_variables_initializer()

	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(num_epochs):

			minibatch_cost = 0.
			num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
			seed = seed + 1
			minibatches = random_mini_batches(x_train, y_train, minibatch_size, seed)

			for minibatch in minibatches:

				                # Select a minibatch
				(minibatch_X, minibatch_Y) = minibatch
				 

				_ , temp_cost = sess.run([optimizer, cost], feed_dict={x: minibatch_X, y: minibatch_Y})
				                
				minibatch_cost += temp_cost / num_minibatches

			if(epoch % 10 == 0):
				print(minibatch_cost)

		predict_op = tf.argmax(z3, 1)
		correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
	        
	        # Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(x_test.shape, y_test.shape)
		train_accuracy = accuracy.eval({x: x_train, y: y_train})
		#test_accuracy = accuracy.eval({x: x_test, y: y_test})
		print("Train Accuracy:", train_accuracy)
		#print("Test Accuracy:", test_accuracy)			
		


	plt.plot(np.squeeze(costs), 'r')
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()



if __name__ == '__main__':

	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

	index = 6
	plt.imshow(X_train_orig[index])
	plt.show()
	print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


	X_train = X_train_orig/255.
	X_test = X_test_orig/255.
	Y_train = convert_to_one_hot(Y_train_orig, 6).T
	Y_test = convert_to_one_hot(Y_test_orig, 6).T

	learning_rate = 0.009
	num_epochs = 10

	train(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs)

