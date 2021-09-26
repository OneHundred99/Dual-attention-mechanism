# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, Add,Dropout,BatchNormalization,MaxPooling3D,concatenate,add,Multiply, multiply
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf


def conv_block(x_in, nf, strides=1):
	"""
	specific convolution module including convolution followed by leakyrelu
	"""
	ndims = len(x_in.get_shape()) - 2
	assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

	Conv = getattr(KL, 'Conv%dD' % ndims)
	x_out = Conv(nf, kernel_size=3, padding='same',
				 kernel_initializer='he_normal', strides=strides)(x_in)
	x_out = LeakyReLU(0.2)(x_out)
	return x_out


def BatchActivate(x):
	x = BatchNormalization()(x)
	#    x = Activation('relu')(x)
	x = LeakyReLU(0.2)(x)
	return x


def res_conv_block(x_in, nf, kernel_size=3, strides=1, activation=True):
	ndims = len(x_in.get_shape()) - 2
	assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

	Conv = getattr(KL, 'Conv%dD' % ndims)
	x = Conv(nf, kernel_size, padding='same',
			 kernel_initializer='he_normal', strides=strides)(x_in)
	if activation == True:
		x = BatchActivate(x)
	return x


def residual_block(blockInput, nf, strides=1, batch_activate=False):
	"""
	specific convolution module including convolution followed by leakyrelu
	"""
	x = BatchActivate(blockInput)
	x = res_conv_block(x, nf)
	x = res_conv_block(x, nf, activation=False)
	x = Add()([x, blockInput])
	if batch_activate:
		x = BatchActivate(x)
	return x


def dense_block(x_in, nf, strides=1):
	"""
	specific convolution module including convolution followed by leakyrelu
	"""
	ndims = len(x_in.get_shape()) - 2
	assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

	Conv = getattr(KL, 'Conv%dD' % ndims)
	x_out = Conv(nf, kernel_size=3, padding='same',
				 kernel_initializer='he_normal', strides=strides)(x_in)
	x_out = LeakyReLU(0.2)(x_out)
	return x_out


def sample(args):
	"""
	sample from a normal distribution
	"""
	mu = args[0]
	log_sigma = args[1]
	noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
	z = mu + tf.exp(log_sigma / 2.0) * noise
	return z





class Sample(Layer):
	"""
	Keras Layer: Gaussian sample from [mu, sigma]
	"""

	def __init__(self, **kwargs):
		super(Sample, self).__init__(**kwargs)

	def build(self, input_shape):
		super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		return sample(x)

	def compute_output_shape(self, input_shape):
		return input_shape[0]


class Negate(Layer):
	"""
	Keras Layer: negative of the input
	"""

	def __init__(self, **kwargs):
		super(Negate, self).__init__(**kwargs)

	def build(self, input_shape):
		super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		return -x

	def compute_output_shape(self, input_shape):
		return input_shape


class Rescale(Layer):
	"""
	Keras layer: rescale data by fixed factor
	"""

	def __init__(self, resize, **kwargs):
		self.resize = resize
		super(Rescale, self).__init__(**kwargs)

	def build(self, input_shape):
		super(Rescale, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		return x * self.resize

	def compute_output_shape(self, input_shape):
		return input_shape


class RescaleDouble(Rescale):
	def __init__(self, **kwargs):
		self.resize = 2
		super(RescaleDouble, self).__init__(self.resize, **kwargs)





class LocalParamWithInput(Layer):
	"""
	The neuron.layers.LocalParam has an issue where _keras_shape gets lost upon calling get_output :(
		tried using call() but this requires an input (or i don't know how to fix it)
		the fix was that after the return, for every time that tensor would be used i would need to do something like
		new_vec._keras_shape = old_vec._keras_shape

		which messed up the code. Instead, we'll do this quick version where we need an input, but we'll ignore it.

		this doesn't have the _keras_shape issue since we built on the input and use call()
	"""

	def __init__(self, shape, my_initializer='RandomNormal', mult=1.0, **kwargs):
		self.shape = shape
		self.initializer = my_initializer
		self.biasmult = mult
		super(LocalParamWithInput, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(name='kernel',
									  shape=self.shape,  # input_shape[1:]
									  initializer=self.initializer,
									  trainable=True)
		super(LocalParamWithInput, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		# want the x variable for it's keras properties and the batch.
		b = 0 * K.batch_flatten(x)[:, 0:1] + 1
		params = K.expand_dims(K.flatten(self.kernel * self.biasmult), 0)
		z = K.reshape(K.dot(b, params), [-1, *self.shape])
		return z

	def compute_output_shape(self, input_shape):
		return (input_shape[0], *self.shape)


class MergeInputs3D(Layer):
	"""
	The MergeInputs3D Layer merge two 3D inputs with Inputs1*alpha+Inputs2*beta, where alpha+beta=1.
	keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
	keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))
	"""

	def __init__(self, output_dim=1, my_initializer='RandomUniform', **kwargs):
		#        self.shape=shape
		self.initializer = my_initializer
		self.output_dim = output_dim
		super(MergeInputs3D, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		if len(input_shape) > 2:
			raise Exception('must be called on a list of length 2.')

		# set up number of dimensions
		#        self.ndims = len(input_shape[0]) - 2
		self.inshape = input_shape
		#        shape_kernel = [self.output_dim]
		#        shape_kernel.append()
		#        shape_kernel1 = tuple(shape_kernel)
		shape_a = input_shape[0][1:]  # input_shape[0][1:-1]
		#        shape_b = input_shape[1][1:]
		#        shape_a, shape_b = input_shape
		#        assert shape_a == shape_b
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='kernel',
									  shape=shape_a,
									  initializer=RandomNormal(mean=0.5, stddev=0.001),
									  #                                      initializer=self.initializer,#'uniform',
									  #                                      constraint = min_max_norm,
									  trainable=True)
		super(MergeInputs3D, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, inputs):
		# check shapes
		assert isinstance(inputs, list)
		assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
		vol1 = inputs[0]
		vol2 = inputs[1]
		batch_size = tf.shape(vol1)[0]
		#        height = tf.shape(vol1)[1]
		#        width = tf.shape(vol1)[2]
		#        depth = tf.shape(vol1)[3]
		#        channels = tf.shape(vol1)[4]
		#        vol1, vol2 = inputs

		# necessary for multi_gpu models...
		#        vol1 = K.reshape(vol1, [-1, *self.inshape[0][1:]])
		#        vol2 = K.reshape(vol2, [-1, *self.inshape[1][1:]])

		alpha = self.kernel  # + 0.5
		alpha = tf.expand_dims(alpha, 0)
		alpha = tf.tile(alpha, [batch_size, 1, 1, 1, 1])
		beta = 1 - alpha
		return vol1 * alpha + vol2 * beta

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		#        shape_a = input_shape[0][1:-1]
		#        shape_b = input_shape[1][1:]
		#        shape_a, shape_b = input_shape
		#        assert shape_a == shape_b
		return input_shape[0]


class NCCLayer(Layer):
	"""
	local (over window) normalized cross correlation
	"""

	def __init__(self, win=None, channels=1, alpha=0.5, eps=1e-5, **kwargs):
		self.win = win
		self.channels = channels
		self.alpha = alpha
		self.eps = eps
		super(NCCLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		if isinstance(input_shape[0], (list, tuple)):
			input_shape = input_shape[0]

		# set up number of dimensions
		self.ndims = len(input_shape) - 2
		self.inshape = input_shape

		# confirm built
		self.built = True

	def call(self, inputs):
		# check shapes
		assert isinstance(inputs, list)
		assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
		I = inputs[0]
		J = inputs[1]
		#        batch_size = tf.shape(vol1)[0]

		# get dimension of volume
		# assumes I, J are sized [batch_size, *vol_shape, nb_feats]
		ndims = len(I.get_shape().as_list()) - 2
		assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

		# set window size
		if self.win is None:
			self.win = [9] * ndims

		# get convolution function
		conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

		# compute CC squares
		I2 = I * I
		J2 = J * J
		IJ = I * J
		# compute filters
		#        sum_filt = tf.ones([*self.win,self.channels, self.channels])
		sum_filt = tf.ones([*self.win, self.channels, self.channels])
		strides = 1
		if ndims > 1:
			strides = [1] * (ndims + 2)
		padding = 'SAME'

		# compute local sums via convolution
		I_sum = conv_fn(I, sum_filt, strides, padding)
		J_sum = conv_fn(J, sum_filt, strides, padding)
		I2_sum = conv_fn(I2, sum_filt, strides, padding)
		J2_sum = conv_fn(J2, sum_filt, strides, padding)
		IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

		# compute cross correlation
		win_size = np.prod(self.win)
		u_I = I_sum / win_size
		u_J = J_sum / win_size

		cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
		I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
		J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

		cc = cross * cross / (I_var * J_var + self.eps)

		# return negative cc.
		#        return cc
		return 1 - tf.reduce_mean(cc)

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		#        shape_a = input_shape[0][1:-1]
		#        shape_b = input_shape[1][1:]
		#        shape_a, shape_b = input_shape
		#        assert shape_a == shape_b
		return (1, 1, 1)


class MyTile(Layer):
	"""
	local (over window) normalized cross correlation
	"""

	def __init__(self, channels=1, **kwargs):
		self.channels = channels
		super(MyTile, self).__init__(**kwargs)

	def build(self, input_shape):
		# set up number of dimensions
		self.inputshape = input_shape
		# confirm built
		self.built = True

	def call(self, inputs):
		# check shapes
		inputs = tf.tile(inputs, [1, 1, 1, 1, self.channels])
		return inputs

	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		assert len(shape) == 5  # only valid for 2D tensors
		shape[-1] *= self.channels
		return tuple(shape)


def Attention_block(up_in, down_in, nf):
	"""
	specific convolution module including convolution followed by leakyrelu
	"""
	#    def MyTile(alpha, channels):
	#        alpha = tf.tile(alpha, [1, 1, 1, 1, channels])
	#        return alpha
	#    def MyTile_output_shape(input_shape):
	#        shape = list(input_shape)
	#        assert len(shape) == 5  # only valid for 2D tensors
	#        shape[-1] *= channels
	#        return tuple(shape)
	ndims = len(up_in.get_shape()) - 2
	assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
	input_channels = up_in.get_shape().as_list()[-1]
	#    batch_size1 = tf.shape(down_in)[0]
	#    nf  = tf.min(batch_size0,batch_size1)
	Conv = getattr(KL, 'Conv%dD' % ndims)
	up = Conv(nf, kernel_size=1, padding='same',
			  kernel_initializer='he_normal', strides=2)(up_in)
	down = Conv(nf, kernel_size=1, padding='same',
				kernel_initializer='he_normal', strides=1)(down_in)

	#    x = NCCLayer(channels=nf)([up,down])
	x = Add()([up, down])
	x = Activation('relu')(x)
	x = Conv(1, kernel_size=1, padding='same',
			 kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', strides=1, activation='sigmoid')(
		x)
	#    x = Activation('sigmoid')(x)
	upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
	alpha = upsample_layer()(x)
	#    alpha = Lambda(MyTile)(alpha, input_channels)
	alpha = MyTile(channels=input_channels)(alpha)
	up_out = Multiply()([alpha, up_in])
	return up_out


def Anatomical_attention_gate(featureMap1, featureMap2):
	"""
	specific convolution module including convolution followed by leakyrelu
	"""
	ndims = len(featureMap1.get_shape()) - 2
	assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
	#    input_channels = featureMap1.get_shape().as_list()[-1]
	#    batch_size1 = tf.shape(down_in)[0]
	#    nf  = tf.min(batch_size0,batch_size1)
	featureMap = concatenate([featureMap1, featureMap2])
	Conv = getattr(KL, 'Conv%dD' % ndims)
	tensorweight1 = Conv(1, kernel_size=1, padding='same',
						 kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', strides=1,
						 activation='sigmoid')(featureMap)
	#    tensorweight1 = Activation('relu')(tensorweight1)
	w_featureMap1 = Multiply()([featureMap1, tensorweight1])
	tensorweight2 = Conv(1, kernel_size=1, padding='same',
						 kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', strides=1,
						 activation='sigmoid')(featureMap)
	#    tensorweight2 = Activation('relu')(tensorweight2)
	w_featureMap2 = Multiply()([featureMap2, tensorweight2])
	w_featureMap = Add()([w_featureMap1, w_featureMap2])
	return w_featureMap