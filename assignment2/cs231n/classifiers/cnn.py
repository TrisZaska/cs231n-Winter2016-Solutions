import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    num_channels = input_dim[0]
    height, width = input_dim[1], input_dim[2]
    self.params['W1'] = weight_scale * np.random.randn(num_filters,
                                                        num_channels,
                                                        filter_size,
                                                        filter_size)
    self.params['b1'] = np.zeros((num_filters, 1))
    self.params['W2'] = weight_scale * np.random.randn(num_filters*(height/2)*(width/2),
                                                        hidden_dim)
    self.params['b2'] = np.zeros((1, hidden_dim))
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim,
                                                        num_classes)
    self.params['b3'] = np.zeros((1, num_classes))
    if self.use_batchnorm == True:
        self.params['gamma1'] = np.ones(num_filters)
        self.params['gamma2'] = np.ones(hidden_dim)
        self.params['beta1'] = np.zeros(num_filters)
        self.params['beta2'] = np.zeros(hidden_dim)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    mode = 'test' if y is None else 'train'
    if self.use_batchnorm:
        gamma1, gamma2 = self.params['gamma1'], self.params['gamma2']
        beta1, beta2 = self.params['beta1'], self.params['beta2']
        self.sp_params = {'mode':mode}
        self.bn_params = {'mode':mode}
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    conv_layer1, cache_layer1 = conv_forward_fast(X, W1, b1, conv_param)
    if self.use_batchnorm:
        conv_layer1, cache_sp_layer1 = spatial_batchnorm_forward(conv_layer1,
                                                                 gamma1,
                                                                 beta1,
                                                                 self.sp_params)
    ReLU1, cache_ReLU1 = relu_forward(conv_layer1)
    max_pool1, cache_maxpool1 = max_pool_forward_fast(ReLU1, pool_param)
    aff2, cache_aff2 = affine_forward(max_pool1, W2, b2)
    if self.use_batchnorm:
        aff2, cache_sp_aff2 = batchnorm_forward(aff2,
                                             gamma2,
                                             beta2,
                                             self.bn_params)
    ReLU2, cache_ReLU2 = relu_forward(aff2)
    aff3, cache_aff3 = affine_forward(ReLU2, W3, b3)
    scores = aff3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    
    grad3 = ReLU2.T.dot(dscores)
    grads['W3'] = grad3 + (self.reg * W3)
    db3 = np.sum(dscores, axis=0)
    grads['b3'] = np.reshape(db3, (1, db3.shape[0]))
    
    dReLU2,_,_ = affine_backward(dscores, cache_aff3)
    daff2 = relu_backward(dReLU2, cache_ReLU2)
    if self.use_batchnorm:
        daff2, dgamma, dbeta = batchnorm_backward_alt(daff2, cache_sp_aff2)
        grads['gamma2'] = dgamma
        grads['beta2'] = dbeta
    
    grad2 = cache_aff2[0].T.dot(daff2)
    grads['W2'] = grad2 + (self.reg * W2)
    db2 = np.sum(daff2, axis=0)
    grads['b2'] = np.reshape(db2, (1, db2.shape[0]))
    
    dmaxpool1,_,_ = affine_backward(daff2, cache_aff2)
    dmaxpool1 = np.reshape(dmaxpool1, max_pool1.shape)
    dReLU1 = max_pool_backward_fast(dmaxpool1, cache_maxpool1)
    dconv_layer1 = relu_backward(dReLU1, cache_ReLU1)
    if self.use_batchnorm:
        dconv_layer1, dgamma, dbeta = spatial_batchnorm_backward(dconv_layer1, cache_sp_layer1)
        grads['gamma1'] = dgamma
        grads['beta1'] = dbeta
    _, dW1, db1 = conv_backward_fast(dconv_layer1, cache_layer1)
    grads['W1'] = dW1 + (self.reg * W1)
    grads['b1'] = np.reshape(db1, (db1.shape[0], 1))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
