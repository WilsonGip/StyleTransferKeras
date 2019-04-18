import numpy as np
import time
import os
import argparse
from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b
from keras.applications import vgg19
from keras import backend as K

parser = argparse.ArgumentParser(description='Artistic Style Transfer with Keras')
parser.add_argument('content_image_path', metavar='content', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_image_path', metavar='style', type=str,
                    help='Path to the style reference image.')
parser.add_argument('results_path_prefix', metavar='res_path_pre', type=str,
                    help='Path and Prefix for the saved results. e.g. \'results/styled\'')
parser.add_argument('--num_iter', type=int, default=50, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=1e0, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1e4, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.5e0, required=False,
					help='Total Variation weight.')
parser.add_argument('--image_prep', type=str, default="content", required=False,
					help="Initial image used to create new styled image. [Default:\"content\",\"noise\"]")
parser.add_argument('--save_every_x', type=int, default=5, required=False,
					help="Save output image at every x iteration. Default: 5")
parser.add_argument('--image_size', type=int, default=512, required=False,
					help="Rescale input and output image. Default: 512")

# Input arguments for the program
args = parser.parse_args()
content_image_path = args.content_image_path
style_image_path = args.style_image_path
result_prefix = args.results_path_prefix
iterations = args.num_iter + 1
image_prep = args.image_prep
every_x = args.save_every_x

# these are the weights of the different loss functions
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# dimensions of the generated picture.
img_width = img_height = args.image_size

# Preprocess the image by resizing and expanding dimension
def preprocess_image(image_path):
	img = imread(image_path, mode="RGB")
	img = imresize(img, (img_width, img_height))
	img = np.expand_dims(img, axis=0)
	return img


# Get the generated image
def deprocess_image(img):
	img = img.reshape((img_width, img_height, 3))
	img = np.clip(img, 0, 255).astype('uint8')
	return img

# Create the VGG19 pretrained model and return a dictionary with all of the VGG19's layers output
def get_outputs_dict():
	content_image = K.variable(preprocess_image(content_image_path))
	style_image = K.variable(preprocess_image(style_image_path))

	input_tensor = K.concatenate(
		[content_image,
		style_image,
		combination_image],
		axis=0)

	model = vgg19.VGG19(
		input_tensor=input_tensor,
		weights=None,
		include_top=False)

	model.load_weights('vgg19_notop.h5')

	for layer in model.layers:
		layer.trainable = False

	return dict([(layer.name, layer.output) for layer in model.layers])

# Return the Gram Matrix
def gram_matrix(x):
	assert K.ndim(x) == 3
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0 ,1)))
	gram = K.dot(features, K.transpose(features))
	return gram

# Style Loss Layer Function
def style_loss_layer(style, combination):
	assert K.ndim(style) == 3
	assert K.ndim(combination) == 3
	style_matrix = gram_matrix(style)
	combine_matrix = gram_matrix(combination)
	size = img_width * img_height
	channels = 3
	return K.sum(K.square(style_matrix - combine_matrix)) / (4.0 * (channels ** 2) * (size ** 2))

# Content Loss Function
def content_loss(content, combination):
	return K.sum(K.square(combination - content))

# Total Variation Function, to reduce noise and smooth out the image.
def total_variation_loss(x):
	assert K.ndim(x) == 4
	a = K.square(
    	x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
	b = K.square(
    	x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

# Style Loss Function
# Iterate through feature_layers from the dictionary and return the total style loss
def style_loss(outputs_dict):
	feature_layers = [
		'block1_conv1',
		'block2_conv1',
		'block3_conv1',
		'block4_conv1',
		'block5_conv1']

	loss = K.variable(0.0)

	for layer in feature_layers:
		layer_features = outputs_dict[layer]
		style_features = layer_features[1, :, :, :]
		combination_features = layer_features[2, :, :, :]
		sloss = style_loss_layer(style_features, combination_features)
		loss += (style_weight / len(feature_layers)) * sloss
	return loss

# The Total Loss Function, which add the weighted losses for content, style, and total variation
def get_total_loss(outputs_dict):
	loss = K.variable(0.0)
	layer_features = outputs_dict['block5_conv2']
	content_image_features = layer_features[0, :, :, :]
	combination_features = layer_features[2, :, :, :]
	loss += content_weight * content_loss(content_image_features, combination_features)
	loss += style_loss(outputs_dict)
	loss += total_variation_weight * total_variation_loss(combination_image)
	return loss

# Create the function to calculate total loss and gradient with K.function
def create_loss_and_grad_func(loss, gradient):
    outputs = [loss]
    if type(gradient) in {list, tuple}:
        outputs += gradient
    else:
        outputs.append(gradient)
    f_outputs = K.function([combination_image], outputs)
    return f_outputs
    
# Initial generated image, either from the original content image or a random noise image
def prepare_image():
	assert image_prep in ["content", "noise"] , "image_prep must be one of ['content', 'noise']"
	if "content" in image_prep:
		x = preprocess_image(content_image_path)
	else:
		x = np.random.uniform(0, 255, (1, img_width, img_height, 3))
	return x

# Function to calculate total loss
def cal_loss(x):
    x = x.reshape((1, img_width, img_height, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    return loss_value

# Function to calculate the gradient
def get_grad(x):
	x = x.reshape((1, img_width, img_height, 3))
	outs = f_outputs([x])
	if len(outs[1:]) == 1:
		grad = outs[1].flatten().astype('float64')
	else:
		grad = np.array(outs[1:]).flatten().astype('float64')
	return grad

combination_image = K.placeholder((1, img_width, img_height, 3))

outputs_dict = get_outputs_dict()

loss = get_total_loss(outputs_dict)

f_outputs = create_loss_and_grad_func(loss, K.gradients(loss, combination_image))

img = prepare_image()

# Iterate through and minimize the loss with Scipy's optimization function fmin_l_bfgs_b
# in other words a minimization function using Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bound algorithm
for i in range(iterations):
	print('Start of iteration', i)
	start_time = time.time()
	img, min_val, info = fmin_l_bfgs_b(cal_loss, img.flatten(),
										fprime=get_grad, maxfun=20)
	print('Current loss value:', min_val)
	img = deprocess_image(img.copy())
	fname = result_prefix + '_at_iteration_%d.png' % i
	if i % every_x == 0:
		imsave(fname, img)
		print('Image saved as', fname)
	end_time = time.time()
	print('Iteration %d completed in %ds' % (i, end_time - start_time))