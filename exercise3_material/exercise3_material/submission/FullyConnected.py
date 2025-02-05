from .Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
  def __init__(self,input_size, output_size):
    super().__init__()
    self.trainable = True
    self.input_size = input_size
    self.output_size = output_size
    self.weights = np.random.rand(input_size + 1 , output_size)
    self.input_tensor_val = None
    self.optimizerval = None
    self.gradient_weightsval = None

  def forward(self, input_tensor):
    self.batchsize, _ = np.shape(input_tensor)
    self.input_tensor_val = np.c_[input_tensor, np.ones(self.batchsize)]
    return self.input_tensor_val @ self.weights    

  def backward(self, error_tensor):
    err_prev = error_tensor@self.weights[:-1,:].T
    self.gradient_weightsval = self.input_tensor_val.T@error_tensor
    if(self.optimizerval != None):
      self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
    return err_prev

  def calculate_update(self, weight_tensor, gradient_tensor):
    if(self.optimizer == None):
      return
    self.optimizer.calculate_update(weight_tensor, gradient_tensor)
  
  @property
  def optimizer(self):
    return self.optimizerval
  
  @optimizer.setter
  def optimizer(self,newval):
    self.optimizerval = newval

  @property
  def gradient_weights(self):
    return self.gradient_weightsval
  
  @gradient_weights.setter
  def gradient_weights(self,newval):
    self.gradient_weightsval = newval

  def initialize(self, weights_initializer, bias_initializer):
    input_dim, output_dim = self.weights.shape
    self.weights[:self.input_size, : ] = weights_initializer.initialize(self.weights[:-1,:].shape, input_dim - 1, output_dim)
    self.weights[-1, :] = bias_initializer.initialize(self.weights[-1,:].shape, input_dim - 1, output_dim)
  
  @property
  def activation(self):
    return self.input_tensor_val
  
  @activation.setter
  def activation(self, new_ten):
    self.input_tensor_val = np.c_[new_ten, np.ones(self.batchsize)]
  