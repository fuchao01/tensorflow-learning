#%matplotlib inline

from matplotlib import pyplot as plt
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("/Users/fuchao/github/models/slim")
import numpy as np

import tensorflow as tf
import urllib2

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

checkpoints_dir = '/Users/fuchao/checkpoints/'

slim = tf.contrib.slim

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = inception.inception_v4.default_image_size
image_channel = 3
lables_map = {}

def get_data():
	images = []
	processed_images_batch = []
	for line in open("/Users/fuchao/pictest.txt"):
		url = line.split(',')[0].strip()
		print url

		request = urllib2.Request(url)
		request.add_header('User-Agent','Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6')
		try:
			response = urllib2.urlopen(request)
		except urllib2.URLError, e:
			print e.code
			continue
		# Open specified url and load image as a string
		image_string = urllib2.urlopen(url).read()
		image = tf.image.decode_image(image_string, channels=3)
		# Resize the input image, preserving the aspect ratio
		# and make a central crop of the resulted image.
		# The crop will be of the size of the default image size of
		# the network.
		#print image
		processed_image = inception_preprocessing.preprocess_image(image,
  	                           image_size,
  	                           image_size,
  	                           is_training=False)
		# Networks accept images in batches.
		# The first dimension usually represents the batch size.
		# In our case the batch size is one.
		processed_images  = tf.expand_dims(processed_image, 0)
		images.append(processed_images)
		processed_images_batch.append(processed_image)
		lables_map[line.split(',')[0].strip()] = line.split(',')[1].strip()
	return images,processed_images_batch

with tf.Graph().as_default():
	
	#url = ("http://img04.tooopen.com/images/20130123/tooopen_09221274.jpg")
	
	# Open specified url and load image as a string
	#image_string = urllib2.urlopen(url).read()
	
	# Decode string into matrix with intensity values
	#image = tf.image.decode_jpeg(image_string, channels=3)
	X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, image_channel])
	images,processed_images = get_data()
	# Create the model, use the default arg scope to configure
	# the batch norm parameters. arg_scope is a very conveniet
	# feature of slim library -- you can define default
	# parameters for layers -- like stride, padding etc.
	with slim.arg_scope(inception.inception_v4_arg_scope()):
		logits, _ = inception.inception_v4(X,
                 num_classes=1001,
                 is_training=False)
	# In order to get probabilities we apply softmax on the output.
	probabilities = tf.nn.softmax(logits)
	
	# Create a function that reads the network weights
	# from the checkpoint file that you downloaded.
	# We will run it in session later.
	#print slim.get_model_variables()
	init_fn = slim.assign_from_checkpoint_fn(
		os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
		slim.get_model_variables('InceptionV4'))
	names = imagenet.create_readable_names_for_imagenet_labels()
	with tf.Session() as sess:
		
		# Load weights
		init_fn(sess)
		#for image,processed_image in images,processed_images:
		for n in xrange(len(images)):
			image,processed_image = images[n],processed_images[n]
			
			image_bytes = sess.run(tf.cast(image, tf.float32))
			image_bytes = np.array(image_bytes).reshape([-1, image_size, image_size, image_channel])
			#print type(image_bytes)
			pro = sess.run([probabilities],feed_dict={X: image_bytes})
			#print type(probabilities)
			#print probabilities
			pro = pro[0]
			pro = pro[0]
			#print type(probabilities)
			#print pro
			sorted_inds = [i[0] for i in sorted(enumerate(-pro),
                       key=lambda x:x[1])]
			for j in range(5):
				index = sorted_inds[j]
				# Now we print the top-5 predictions that the network gives us with
				# corresponding probabilities. Pay attention that the index with
				# class names is shifted by 1 -- this is because some networks
				# were trained on 1000 classes and others on 1001. VGG-16 was trained
				# on 1000 classes.
				print('n: Probability %d %0.2f => [%s]' % (n,pro[index], names[index+1]))
			print "###########"
	res = slim.get_model_variables()