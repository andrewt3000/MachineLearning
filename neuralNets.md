### Neural Networks
**Neural networks** are machine learning models and [universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem). This document explains their architecture and how to [train a neural network](#training-a-neural-network).    

Neural networks or artificial neural networks are a broad term that includes also other types of neural networks such as CNNs or transformers. The architecture we are discussing here is a feedforward neural network (as opposed to recurrent), vanilla neural networks, or MLP multilayer perceptron.  

You can also refer to a single layer or block of **fully connected layers** (also called **dense layers** or **linear layers**) in other types of neural networks. These fully connected layers are used in other neural networks to combine features or change dimensionality.  
pytorch: [nn.Linear()](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)  

### Neural network Architecture
The architecture of a neural network is fixed before it is trained and has the following properties. 
- Neural networks are composed of input layers, hidden layers and output layers.  
- Layers are connected by weighted synapsis (the lines with arrows) that multiply their input times the weight. 
- Hidden layer consists of neurons (the circles) that sum their inputs from synapsis and execute an activation function on the sum.  
- Neural networks also typically have a single bias input node that is a constant value. It's similar to the constant in a linear function. Biases ensure that even when all input features are zero, a neuron can still output a non-zero value. (The bias is missing in diagram below)  
The weights and biases are often refered to as parameters.   

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/nn.png" height='250px' width='250px'/>  

pytorch layers are compose in [Sequential()](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) containers.

### Hyperparameters
**Hyperparameters** are the model’s parameters in a neural net such as architecture, learning rate, and regularization factor.	

**Architecture** is the structure of a neural network i.e. number of hidden layers, and number of nodes. 

**Number of hidden layers** is a hyperparmeter. The higher the number of layers the more layers of abstraction the network can represent. If the network has too many layers it may suffer from the vanishing or exploding gradient problem.  

**Capacity** is the model's storage space for patterns and is driven by parameter width (number of hidden units) and depth (number of layers). Generally, a higher number of parameters equates to a higher capacity.   

### Activation Functions
**Activation function** - the "neuron" in the neural network executes an activation function on the sum of the weighted inputs. In the neuron metaphor you can assume as the value approaches 1 the neuron is "firing". ReLu is a popular modern activation function.  
pytorch [activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)  

#### ReLu
ReLu activation is currently popular in linear layers and cnns. ReLu stands for rectified linear unit. It returns 0 for negative values, and the same number for positive values. Relu can suffer from "dead" relus ()    
pytorch [nn.ReLU()](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html)  

```python
def relu(x):
  if x < 0:
    return 0
  if x >= 0:
    return x
```

#### GELU
GELU is popular in transformers. GELU stands for Gaussian Error Linear Units function.  
pytorch [nn.GELU()](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)

#### Sigmoid
Sigmoid activation functions outputs a value between 0 and 1. It is a smoothed out step function. Sigmoid is not zero centered and it suffers from activation saturation issues. Historically popular, but not currently popular. Might be used for binary classification.  
pytorch [nn.Sigmoid()](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)

#### Softmax
The softmax function is often used as the model's final output activation function for multi-class classification. The output is similar to a probability distribution accross the labels however it's a point of debate if it should be consider a probability distribution in the freuqentist sense. Softmax is a "soft" maximum function. It's properties are:  
Output values are in the range [0, 1].  
The sum of output nodes is 1.  

The softmax function as applied to each node NN output is the exponent of the output divided by the sum of all the exponent outputs. So for instance, if there are 3 nodes, the output of the 1st node y1 is:     
e <sup>y^1</sup> / (e <sup>y^1</sup> + e <sup>y^2</sup> + e <sup>y^3</sup>)

```python
def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)
```
pytorch [nn.Softmax()](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)  

### Training a neural network
Training a neural network - minimize a cost function. Use backpropagation and gradient descent to adjust weights to make model more accurate. 

Steps to training a network.  
- [Prepare the data](#prepare-the-data)
- [initialize weights and biases](#initialization)  
- [Implement forward propagation](#forward-propagation)  
- [Implement loss function](#loss-function)
- [Implement backpropagation](#backpropagation) 
- [Run optimization algorithm](#optimization-algorithms) 

Training is typically implemented as a loop where each loop is an epoch. An **epoch** represents one complete pass of the entire training dataset through the neural network. On each iteration of the loop a loss function is calculated and backpropagation is performed. This is repeated until the decision to terminate is reached.  

### Prepare the data
Begin by preparing and scaling the data. See [section on data and features](https://github.com/andrewt3000/MachineLearning/blob/master/README.md#data).  


### Initialization
The weights are historically initialized with small random numbers centered on zero. If the weights are the same (say all 0s) they will remain the same throughout training, making the weights random breaks this symmetry (Rumelhart et. al 1986). Bias is typically initialized to 0.     

As your neural networks get deeper, initialization becomes more important. If the initial weights are too small, you get a vanishing gradient. If the initial weights are too large, you get an exploding gradient. 

- Xavier (Glorot) Initializations are typically used on symmetric activation functions like Tanh or Sigmoid.  
- Kaiming (He) initializations are typically used on non-linear activations like ReLU or LeakyReLU

### Forward Propagation
The forward propagation function is called at inference. The input is a vector of the features X and the output returned is a vector of the values after traversing the network.  

#### numpy Example
If X is the input matrix, and W1 is the weight matrix (initialized and trained outside of this scope) for the first hidden layer, we take the dot product to get the values passed to the activation functions. Then we apply the activation function to each element in the matrix. Repeat for each layer.  

```python
def forward(self, X):
  z2 = np.dot(X, self.W1)
  a2 = activation(z2)
  return a2
```

[Example of Forward propagation in numpy](https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/.ipynb_checkpoints/Part%202%20Forward%20Propagation-checkpoint.ipynb)

#### Pytorch Example 
In pytorch, implement the forward() method of the [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) class  

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        # Passing the linear layer output through a ReLU activation
        x = F.relu(self.fc(x)) 
        return x

```

### Loss Function 
The next step is to choose a loss function. Then implement the loss function or use a loss function from an existing library. The loss function measures how inaccurate a model is for a single example. Training a model minimizes the loss function. Mean squared error is a typical loss function for regression. Cross entropy is a typical loss function for classification.   

| ML Problem | Loss Function | PyTorch Class |
| :--- | :--- | :--- |
| **Regression** | Mean Squared Error (MSE) | [`nn.MSELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) |
| **Classification** | Cross Entropy | [`nn.CrossEntropyLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |


- The term loss function applies to a single example.  
- The term error function refers to a single example and whether it's right or wrong for performance measurment, not training.  
- The terms cost function, objective function, and total loss refer to the entire dataset or mini-batch and may also include regularization in addition to the sum of the loss.  

pytorch [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) 

#### Cross entropy
Cross entropy (aka log loss, negative log probability) function is frequently used with classification models that use a softmax activation function for the output layer. The output of the softmax activation is a value between 0 and 1. The loss will be 0 if the output value is 1 and that is the correct classification. Conversely, the loss approaches infinity as the output approaches 0 for the correct classification.  


<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/cross_entropy.png" />

### numpy example
```python
def logloss(true_label, predicted_prob):
  if true_label == 1:
    return -log(predicted_prob)
  else:
    return -log(1 - predicted_prob)
 ```



### Backpropagation 
The backpropagation algorithm applies the chain rule recursively to compute the gradient for each weight. The gradient is caulated by taking the partial derivative of the loss function with respect to the weights at each layer of the network by moving backwards (output to input) through the network. Backprop indicates how to adjust the weights to minimize the loss function. If the gradient (i.e. partial derivative/slope) is positive, that means the loss is getting higher as the weight increases. If the derivative is 0, the weight is set to a minimum loss. The gradient indicates the magnitude and direction of adjustments to our weights that will reduce the loss.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/descent.png"   height='360px' width='640px' />

We can combine all these functions for cost and forward propagation to get one function. So for instance, the cost function for a NN with one hidden relu layer using a softmax output is (where W1 and W2 are weights for 1st and 2nd layers, and X1 is the input feature nodes:  
  
```
J = Cost(Softmax(DotProduct(Relu(DotProduct(X1,W1)), W2)))
```

We then calculate the partial derivative of the loss function with respect to each weights (dJ/dW). We use the chain rule. The chain rule is the derivative of f(g(x)) is f'(g(x)) * g'(x)

The result is a gradient for each set of weights, dJ/dW1 and dJ/dW2 which are the same size as W1, W2.  

The derivative of the softmax cost function is the probablity for the incorrect labels and the probablity - 1 for the correct label. 

The derivative of the relu function is:

```
def reluprime(x):
    if x > 0: #the derivative dy/dx of y = x is dy/dx = 1 from the power rule in calculus.
        return 1
    else: #The derivative of a constant is zero. if x <0, y = 0 so dy/dx = 0
        return 0
```  
  
Here is an [example of backprop in numpy](https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/.ipynb_checkpoints/Part%204%20Backpropagation-checkpoint.ipynb) for a regression problem that uses sum of squared errors as a cost function and sigmoid activations.  

pytorch [backward()](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)  

#### Learning Rate
Learning rate (&alpha;) - controls the size of the adjustments made during the training process. Typical values are .1, .01, .001. Consider these values are relative to your input features which are typically scaled to ranges such as 0 to 1, or -1 to +1.  
if &alpha; is too low, convergance is slow.
if &alpha; is too high, there is no convergance, because it overshoots the local minimum.  
The learning rate is often reduced to a smaller number over time. This is often called annealing or decay. (examples: step decay, exponential decay)  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/lr.jpg" />



### Optimization algorithms
**Gradient descent** is an iterative optimization algorithm that, in the context of neural networks, adjusts the weight by learning rate times the negative of the gradient (calculated by backpropagation) to mimimize the loss function.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/gd.jpg"  height='360px' width='640px' />

**Batch gradient descent** - The term batch refers to the fact it uses the entire dataset to make one gradient step. Batch works well for small datasets that have convex loss functions. The loss function needs to be convex or it may find a local minimum.    

**Stochastic gradient descent** (sgd) is a variation of gradient descent that uses a single randomly choosen example to make an update to the weights. sgd is more scalable than batch graident descent and is used more often in practice for large scale deep learning. It's random nature makes it unlikely to get stuck in a local minima.  

**Mini batch gradient descent**: Stochastic gradient descent that considers more than one randomly choosen example before making an update. Batch size is a hyperparmeter that determines how many training examples you consider before making a weight update. Typical values are factors of 2, such as 32 or 128. Values are typically in the range of 32-512.  Larger batches are faster to train, but can cause overfitting and require more memory.  Lower batch sizes are the opposite: slower to train, more regularized, and require less memory.  

#### Gradient Descent Optimization
Momentum sgd is a variation that accelerates sgd, dampens oscillations, and helps skip over local minima and saddlepoints. It collects data on each update in a velocity vector to assist in calculating the gradient. The velocity matrix represents the momentum. Rho is a hyperparameter that represents the friction. Rho is in the range of 0 to 1. Typical values for rho are 0.9 and 0.99. Nesterov accelerated gradient descent is a variation that builds on moment and adds a look ahead step.  

Momement sgd is popular for vanilla neural networks. Adam with weight decay is popular with transformer models.  
pytorch [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)  

Other optimization algorithms include: AdaGrad, AdaDelta, Adam, Adamax, NAdam, RMSProp, and AMSGrad.  
pytorch [optimizers](https://pytorch.org/docs/stable/optim.html#algorithms)  

In pytorch, the optimizer's [step()](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html) method updates the model.  




#### Regularization
Underfitting - output doesn't fit the training data well.  
Overfitting - output fits training data well, but doesn't work well on validation or test data.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/over_under.png"/>

Regularization - a technique to minimize overfitting.  

L1 regularization uses sum of absolute value of weights. L1 works best with sparse outputs.  
L2 regularization uses sum of squared weights. L2 doesn't work well with yielding sparse outputs.    

### Dropout
**Dropout** is a form of regularization. "The key idea is to randomly drop units (along with their connections) from the neural network during training." Typical hyperparameter value is .5 (50%). As dropout value approaches zero, dropout has less effect, as it approaches 1 there are more connections are being zeroed out. The remaining active connections are scaled up to compensate for the zeroed out connections. Dropout is typically implemented in training but not present in inference. Dropout is typically applied to units in the hidden layers.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/dropout.png" />  
<sub> <a href="https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf">Dropout: A Simple Way to Prevent Neural Networks from
Overfitting</a> - Srivastava et al 2014 </sub>

### Early termination
**Early termination** is to stop training when the training error is getting lower but the validation error is increasing. This indicates overfitting.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/early_term.png" />


### Other resources
[Neural Networks demystified video](https://www.youtube.com/watch?v=bxe2T-V8XRs) - videos explaining neural networks. Includes [notes](https://github.com/stephencwelch/Neural-Networks-Demystified).    

[Stanford CS231n](http://cs231n.github.io/neural-networks-1/)  

[TensorFlow Neural Network Playground](http://playground.tensorflow.org)  - This demo lets you run a neural network in your browser and see results graphically. I wrote about the lessons on [intuition about deep learning](https://medium.com/@andrewt3000/understanding-tensorflow-playground-c20cdb7a250b).   

[A Recipe for Training Neural Networks - Andrej Karpathy](http://karpathy.github.io/2019/04/25/recipe/) 
