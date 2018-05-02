import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from keras import backend as K
K.set_image_data_format('channels_first')

from fr_utils import *
from inception_blocks_v2 import *


def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist) ,alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)

    
    return loss


def verify(image_path, identity, database, model):
	

	encoding = img_to_encoding(image_path, model)
    
    #Compute distance with identity's image (≈ 1 line)
	dist = np.linalg.norm(encoding - database[identity])
    
    #Open the door if dist < 0.7, else don't open (≈ 3 lines)
	if dist < 0.89:
		print("It's " + str(identity))
		door_open = True
	else:
		print("It's not " + str(identity))
		door_open = False
        
	return dist, door_open


def recognize(image_path, database, model):

	encoding = img_to_encoding(image_path, model)

	min_dist = 100

	for (name, db_enc) in database.items():

		dist = np.linalg.norm(encoding - db_enc)

		if(dist < min_dist):
			min_dist = dist
			identity = name

	if(min_dist > 0.9):
		print("not recognized", min_dist)
	else:
		print("it's " + str(identity), min_dist)			



if __name__ == '__main__':

	FRmodel = faceRecoModel(input_shape=(3, 96, 96))

	FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
	load_weights_from_FaceNet(FRmodel)

	with tf.Session() as test:
	    tf.set_random_seed(1)
	    y_true = (None, None, None)
	    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
	              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
	              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
	    loss = triplet_loss(y_true, y_pred)
	    
	    print("loss = " + str(loss.eval()))


	database = {}

	database["emma"] = img_to_encoding("images/emma1_reshape.jpeg", FRmodel)
	database["joey"] = img_to_encoding("images/joey2_reshape.jpeg", FRmodel)
	

	emma4 = misc.imread("images/emma4.jpg")
	    
	verify("images/joey1_reshape.jpeg", "emma", database, FRmodel)
	recognize("images/joey1_reshape.jpeg", database, FRmodel)

	plt.imshow(emma4)
	plt.show()


