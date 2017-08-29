import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(X.shape[0]):
    scores = X[i].dot(W)
    
    #Avoid exp unstable
    scores -= np.max(scores)
    
    correct_class_scores = scores[y[i]]
    
    probs = np.exp(correct_class_scores) / np.sum(np.exp(scores))
    loss += -np.log(probs)
    #Gradient
    g_probs = np.exp(scores) / np.sum(np.exp(scores), axis=0, keepdims=True)
    g_probs[y[i]] -= 1
    for j in xrange(dW.shape[1]):
        dW[:, j] += g_probs[j] * X[i]
        
    
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W**2)
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW /= X.shape[0]
  dW += reg * W
    
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  mask = np.arange(scores.shape[0])
  #Avoid numerical unstable
  scores[mask] -= np.max(scores, axis=1, keepdims=True)
  
  correct_class_scores = scores[mask, y][:, np.newaxis]
  probs = np.exp(correct_class_scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  loss += np.sum(-np.log(probs))
  
  #Gradient
  g_probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  g_probs[mask,y] -= 1
  dW = X.T.dot(g_probs)
    
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W**2)
    
  dW /= X.shape[0]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

