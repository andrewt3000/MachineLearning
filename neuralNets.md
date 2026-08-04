# Neural Networks
**Neural networks** are machine learning models and universal approximators [3](#references). This document explains their architecture and how to [train a neural network](#training-a-neural-network).    

### Terminology
Neural networks are a broad term that also includes other types of neural networks such as [CNNs](cnn.md) or [transformers](transformer.md). The architecture we are discussing is a vanilla neural network, a directed acyclical graph (DAG), that that also goes by different names:
- **ANN artificial neural networks** as opposed to biological 
- **FNN feedforward neural network** as opposed to recurrent
- **MLP multilayer perceptron** a reference to the original design that inspired neural networks [1](#references)

You can also refer to a single layer or block of **fully connected layers** (also called **dense layers** or **linear layers**) in other types of neural networks. Fully connected layers are used in more complex neural networks, such as [CNNs](cnn.md) or [transformers](transformer.md) to combine features or change dimensionality.  

### PyTorch
PyTorch [nn.Linear()](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)  layers are compose in [Sequential()](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) containers. 


### Neural network architecture
A neural network’s **architecture** is its structural blueprint — the fixed layout of layers, nodes, and connections defined before training begins.
- Neural networks are composed of layers.
  - **Input layers** receive the raw features.
  - **Hidden layers** are the processing engine. A network can have one hidden layer (shallow) or dozens (deep).
  - **Output layers** generate **predictions**.
- The lines connecting nodes between layers represent **weights**. Each connection multiplies the incoming value $x$ by its weight $w$.
- Each node (circle) in a hidden layer performs two sequential steps:
  - Sums all incoming weighted signals
  - Passes that sum through an **[activation function](#activation-functions)** (e.g., ReLU, Sigmoid). This adds non-linearity, allowing the network to learn complex, non-linear patterns rather than simple straight lines. 
- Every hidden and output node has a **bias** term added to its weighted sum.
  - A bias term is similar to the constant in a linear function in that it controls the intercept.
  - Typically turn off bias term if using batch normalization.  
  - The bias is missing in diagram below  
- The equation for a single layer is output Y is the activation sigma of (weights W times input X  + b bias).  $$Y = \sigma(XW + b)$$
- The input vector X is multiplied by the weight vector W into a sum by the [dot product](la.md#dot-product).  
  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/nn.png" height='250px' width='250px'/>  

**Parameters** are the learnable weights and biases.   

**Hyperparameters** are configuration choices set before training the model. Examples include architecture, learning rate, and regularization factor. 

**Number of hidden layers** is a hyperparameter. The higher the number of layers the more layers of abstraction the network can represent. If the network has too many layers it may suffer from the vanishing or exploding gradient problem.  

**Capacity** is the model's storage space for patterns and is driven by parameter width (number of hidden units) and depth (number of layers). Generally, a higher number of parameters equates to a higher capacity.   

**Parameter count** is the total number of learnable weights and biases in a network — every number that gets updated by gradient descent.

### Activation Functions
**Activation function** - Each node executes an activation function on the sum of its weighted inputs. The activation function is what makes a neural network non-linear: without it, stacked layers collapse into a single linear transformation, and the universal approximation theorem [3](#references) requires a non-linear activation. Below are typical activation functions.
**PyTorch** [activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)  

#### ReLU
ReLU activation is currently popular in linear layers and cnns. ReLU stands for rectified linear unit. It returns 0 for negative values, and the same number for positive values. RelU can suffer from "dead" ReLUs ()    
**PyTorch** [nn.ReLU()](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html)  

```python
def relu(x):
  if x < 0:
    return 0
  if x >= 0:
    return x
```

#### GELU
GELU is popular in transformers. GELU stands for Gaussian Error Linear Units function.  
**PyTorch** [nn.GELU()](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)

#### Sigmoid
Sigmoid activation functions outputs a value between 0 and 1. It is a smoothed out step function. Sigmoid is not zero centered and it suffers from activation saturation issues. Historically popular, but not currently popular. Might be used for binary classification.  
**PyTorch** [nn.Sigmoid()](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)

#### Softmax
The softmax function is often used as the model's final output activation function for multi-class classification. The output is similar to a probability distribution across the labels however it's a point of debate if it should be consider a probability distribution in the frequentist sense. Softmax is a "soft" maximum function. Its properties are:  
Output values are in the range [0, 1].  
The sum of output nodes is 1.  

The softmax function as applied to each node NN output is the exponent of the output divided by the sum of all the exponent outputs. So for instance, if there are 3 nodes, the output of the 1st node y1 is:     
e <sup>y^1</sup> / (e <sup>y^1</sup> + e <sup>y^2</sup> + e <sup>y^3</sup>)

```python
def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)
```
**PyTorch** [nn.Softmax()](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)  

### Training a neural network
Training a neural network is an iterative process and the goal is to minimize a cost function. Training is typically implemented as a loop where each loop is an epoch. An **epoch** represents one complete pass of the entire training dataset through the neural network. On each iteration of the loop a forward pass is made then the loss function is calculated, the loss is used with backpropagation to calculate the gradients. Then gradient descent is performed to adjust the weights of the model to minimize the error. This is repeated until the decision to terminate is reached.  

Steps to training a network.  
- [Prepare the data](#prepare-the-data)
- [Initialize weights and biases](#initialization)  
- [Implement forward propagation](#forward-propagation)  
- [Implement loss function](#loss-function)
- [Implement backpropagation](#backpropagation) 
- [Run optimization algorithm](#optimization-algorithms)  

### Prepare the data
Begin by preparing and scaling the data. See [data and features](data.md).  


### Initialization
The weights are historically initialized with small random numbers centered on zero. If the weights are the same (say all 0s) they will remain the same throughout training, making the weights random breaks this symmetry (Rumelhart et. al 1986). Bias is typically initialized to 0.     

As your neural networks get deeper, initialization becomes more important. If the initial weights are too small, you get a vanishing gradient. If the initial weights are too large, you get an exploding gradient. 

- Xavier (Glorot) Initializations are typically used on symmetric activation functions like Tanh or Sigmoid.  
- Kaiming (He) initializations are typically used on non-linear activations like ReLU or LeakyReLU

### Forward Propagation
The forward propagation function is called during training and its output is tested for loss. To implement the forward pass function is implicitly to design the neural network architecture. The forward pass function is also called at inference. The input is a vector of the features X and the output returned is a vector of the values after traversing the network.  

#### Numpy example
If X is the input vector, and W1 is the weight vector (initialized and trained outside of this scope) for the first hidden layer, we take the dot product to get the values passed to the activation functions. Then we apply the activation function to each element in the matrix. Repeat for each layer.  

```python
class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
```

[Example of Forward propagation in numpy](https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/.ipynb_checkpoints/Part%202%20Forward%20Propagation-checkpoint.ipynb)

#### PyTorch example 
In PyTorch, implement the forward() method of the [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) class  

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
The next step is to choose a loss function, either implemented from scratch or from an existing library. The choice of loss function depends on the problem you are trying to solve: mean squared error is typical for regression, cross entropy for classification. Training minimizes the loss function.

- The term **loss function** measures how wrong the model is on a single example. **Error function** is generally a synonym (e.g. "sum of squared errors").
- The terms **objective function** or **cost function** refer to the average (or sum) of the loss over the entire dataset or mini-batch, and may also include a regularization term.
- In some advanced cases, there may be more than one loss function and the sum of the loss functions is referred to as the total loss.  
- Loss functions are distinct from **metrics** such as accuracy or error rate.


#### Cross entropy
Cross entropy (aka log loss, negative log probability) function is typically used as a loss function with classification models that use a softmax activation function for the output layer. The output of the softmax activation is a value between 0 and 1. 

The cross entropy function makes sense intuitively. Consider the case where the label is 1 and the output is 1, the loss is 0 i.e. there is no loss, the output of 1 is correct and equals the label of 1. Conversely, the loss approaches infinity for the output of 0, the incorrect classification.  


<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/cross_entropy.png" />

#### numpy example
```python
def logloss(true_label, predicted_prob):
  if true_label == 1:
    return -log(predicted_prob)
  else:
    return -log(1 - predicted_prob)
 ```

#### PyTorch 
List of pytorch [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) 


| ML Problem | Loss Function | PyTorch Class |
| :--- | :--- | :--- |
| **Classification** | Cross Entropy | [`nn.CrossEntropyLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| **Regression** | Mean Squared Error (MSE) | [`nn.MSELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) |


### Backpropagation 
The backpropagation algorithm applies the chain rule recursively to compute the gradient for each weight. The gradient is calculated by taking the partial derivative of the loss function with respect to the weights at each layer of the network by moving backwards (output to input) through the network. Backprop indicates how to adjust the weights to minimize the loss function. If the gradient (i.e. partial derivative/slope) is positive, that means the loss is getting higher as the weight increases. If the derivative is 0, the weight is set to a minimum loss. The gradient indicates the magnitude and direction of adjustments to our weights that will reduce the loss.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/descent.png"   height='360px' width='640px' />

We can combine all these functions for cost and forward propagation to get one function. So for instance, the cost function for a NN with one hidden ReLU layer using a softmax output is (where W1 and W2 are weights for 1st and 2nd layers, and X1 is the input feature nodes:  
  
```
J = Cost(Softmax(DotProduct(Relu(DotProduct(X1,W1)), W2)))
```

We then calculate the partial derivative of the loss function with respect to each weights (dJ/dW). We use the chain rule. The chain rule is the derivative of f(g(x)) is f'(g(x)) * g'(x)

The result is a gradient for each set of weights, dJ/dW1 and dJ/dW2 which are the same size as W1, W2.  

The derivative of the softmax cost function is the probability for the incorrect labels and the probablity - 1 for the correct label. 

The derivative of the ReLU function is:

```
def reluprime(x):
    if x > 0: #the derivative dy/dx of y = x is dy/dx = 1 from the power rule in calculus.
        return 1
    else: #The derivative of a constant is zero. if x <0, y = 0 so dy/dx = 0
        return 0
```  
  
Here is an [example of backprop in numpy](https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/.ipynb_checkpoints/Part%204%20Backpropagation-checkpoint.ipynb) for a regression problem that uses sum of squared errors as a cost function and sigmoid activations.  

#### PyTorch
In PyTorch the details of backpropagation are abstracted in the function [backward()](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)

#### Learning Rate
Learning rate (&alpha;) - controls the size of the adjustments made during the training process. Typical values are .1, .01, .001. Consider these values are relative to your input features which are typically scaled to ranges such as 0 to 1, or -1 to +1.  
if &alpha; is too low, convergence is slow.
if &alpha; is too high, there is no convergence, because it overshoots the local minimum.  
The learning rate is often reduced to a smaller number over time. This is often called annealing or decay. (examples: step decay, exponential decay)  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/lr.jpg" />



### Optimization algorithms
**Gradient descent** is an iterative optimization algorithm that, in the context of neural networks, adjusts the weight by learning rate times the negative of the gradient (calculated by backpropagation) to minimize the loss function.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/gd.jpg"  height='360px' width='640px' />

**Batch gradient descent** - The term batch refers to the fact it uses the entire dataset to make one gradient step. Batch works well for small datasets that have convex loss functions. The loss function needs to be convex or it may find a local minimum.    

**Stochastic gradient descent** (sgd) is a variation of gradient descent that uses a single randomly chosen example to make an update to the weights. sgd is more scalable than batch gradient descent and is used more often in practice for large scale deep learning. Its random nature makes it unlikely to get stuck in a local minima.  

**Mini batch gradient descent**: Stochastic gradient descent that considers more than one randomly chosen example before making an update. Batch size is a hyperparameter that determines how many training examples you consider before making a weight update. Typical values are factors of 2, such as 32 or 128. Values are typically in the range of 32-512.  Larger batches are faster to train, but can cause overfitting and require more memory.  Lower batch sizes are the opposite: slower to train, more regularized, and require less memory.  

#### Gradient Descent Optimization
Momentum sgd is a variation that accelerates sgd, dampens oscillations, and helps skip over local minima and saddlepoints. It collects data on each update in a velocity vector to assist in calculating the gradient. The velocity matrix represents the momentum. Rho is a hyperparameter that represents the friction. Rho is in the range of 0 to 1. Typical values for rho are 0.9 and 0.99. Nesterov accelerated gradient descent is a variation that builds on moment and adds a look ahead step.  

Momement sgd is popular for vanilla neural networks. Adam with weight decay is popular with transformer models.  

Other optimization algorithms include: AdaGrad, AdaDelta, Adam, Adamax, NAdam, RMSProp, and AMSGrad.  

#### PyTorch
PyTorch list of [optimizers](https://pytorch.org/docs/stable/optim.html#algorithms)  
[SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)    

In PyTorch, the optimizer's [step()](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html) method updates the model. 

Here is a pseudocode example that pulls together the forward pass, loss function (MSELoss), backprop, and the optimizer(SGD) all in an outer training loop that represents an epoch and an inner loop that represents a batch.   

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup: Model, Loss Function, and Optimizer
model = MyNeuralNetwork()
criterion = nn.MSELoss()  # or nn.CrossEntropyLoss() depending on your task
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. Training Loop
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        
        # Forward pass: Compute predicted outputs by passing inputs to the model
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear existing gradients from the last step
        loss.backward()        # Compute gradients of the loss w.r.t. model parameters
        optimizer.step()       # Update model weights based on the computed gradients


```
### Bias and Variance
**Bias** is the error introduced by approximating a complex real-world problem with a model that is too simple.

**Variance** is the error caused by a model's sensitivity to small fluctuations or noise in the training set.

**Underfitting** - output doesn't fit the training data well. (high bias).  

**Overfitting** - output fits training data well, but doesn't work well on validation or test data. (high variance)   

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/over_under.png"/>

**Regularization** is a technique to prevent overfitting.  

L1 regularization uses sum of absolute value of weights. L1 works best with sparse outputs.  
L2 regularization uses sum of squared weights. L2 doesn't work well with yielding sparse outputs.    

### Dropout
**Dropout** is a form of regularization. "The key idea is to randomly drop units (along with their connections) from the neural network during training." Typical hyperparameter value is .5 (50%). As dropout value approaches zero, dropout has less effect, as it approaches 1 there are more connections are being zeroed out. The remaining active connections are scaled up to compensate for the zeroed out connections. Dropout is typically implemented in training but not present in inference. Dropout is typically applied to units in the hidden layers.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/dropout.png" />  
<sub> <a href="https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf">Dropout: A Simple Way to Prevent Neural Networks from
Overfitting</a> - Srivastava et al 2014 </sub>

### Early Stopping
**Early stopping** is a regularization technique. Early stopping is to stop training when the training error is getting lower but the validation error is increasing. Testing poorly on the validation set indicates overfitting. Early stopping is typically based on the loss function (not accuracy).  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/early_term.png" />

### Metrics
**Metrics** measure model performance for evaluation and reporting. Unlike loss functions, metrics don't need to be differentiable because no gradients are computed from them. A model typically trains on one function (e.g. cross entropy) and is reported on another (e.g. accuracy).

Common classification metrics:
- **Accuracy** is the fraction of predictions that are correct. Accuracy is misleading on imbalanced datasets: a model that always predicts "no fraud" is 99% accurate if 1% of transactions are fraudulent.
- **Precision** is the fraction of positive predictions that are actually positive. Precision answers: "when the model says yes, how often is it right?"
- **Recall** is the fraction of actual positives the model finds. Recall answers: "of all the real positives, how many did the model catch?"
- **F1 score** is the harmonic mean of precision and recall, useful when you need a single number that balances both.
- A **confusion matrix** is a table of predicted vs actual classes that shows exactly where the model's mistakes are (false positives vs false negatives).
- **ROC-AUC** measures how well the model ranks positives above negatives across all classification thresholds. 0.5 is random guessing, 1.0 is perfect ranking.

Precision and recall trade off against each other via the **classification threshold** (default 0.5). Lowering the threshold catches more positives (higher recall) at the cost of more false alarms (lower precision).

Common regression metrics include MSE, RMSE, MAE, and R².

scikit-learn [metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

### Calibration
A model is **calibrated** if its predicted probabilities match observed frequencies: among all predictions of 70%, the event should occur about 70% of the time. Calibration is distinct from accuracy — a model can rank outcomes correctly (high accuracy, high AUC) while its probabilities are systematically too confident or not confident enough. Calibration matters whenever the probability itself drives a decision, such as medical risk, bet sizing, or pricing, rather than just the argmax label.

- A **reliability diagram** plots predicted probability (binned) against observed frequency. A calibrated model tracks the diagonal; a curve below the diagonal indicates overconfidence.
- **Expected calibration error (ECE)** summarizes the reliability diagram as a single number: the weighted average gap between predicted probability and observed frequency across bins.
- **Brier score** is the mean squared error between predicted probabilities and outcomes (0 or 1).
- **Proper scoring rules** such as log loss and Brier score are minimized only when predicted probabilities equal the true probabilities, so they reward calibration. Accuracy is not a proper scoring rule — it ignores probability magnitudes entirely.

Modern neural networks are often overconfident even when accurate. Post-hoc fixes include **temperature scaling** (dividing logits by a constant T fitted on the validation set), **Platt scaling**, and **isotonic regression**.

scikit-learn [calibration](https://scikit-learn.org/stable/modules/calibration.html), [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)  


### References
1.  1958 Rosenblatt perceptron paper [THE PERCEPTRON: A PROBABILISTIC MODEL FOR
INFORMATION STORAGE AND ORGANIZATION
IN THE BRAIN](https://homepages.math.uic.edu/~lreyzin/papers/rosenblatt58.pdf)
2. 1986 Rumelhart backprop paper [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0)
3. 1989 Cybenko universal approximation theorem paper - Approximation by superpositions of a sigmoidal function
4. 2012 Alexnet paper [ImageNet Classification with Deep Convolutional
Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
5. 2014 Srivastava Dropout paper [Dropout: A Simple Way to Prevent Neural Networks from
Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
6. 2017 Guo et al. calibration paper [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
### Tutorials, demos
[Neural Networks demystified video](https://www.youtube.com/watch?v=bxe2T-V8XRs) - videos explaining neural networks. Includes [notes](https://github.com/stephencwelch/Neural-Networks-Demystified).    

[TensorFlow Neural Network Playground](http://playground.tensorflow.org)  - This demo lets you run a neural network in your browser and see results graphically. I wrote about the lessons on [intuition about deep learning](https://medium.com/@andrewt3000/understanding-tensorflow-playground-c20cdb7a250b).   

[A Recipe for Training Neural Networks - Andrej Karpathy](http://karpathy.github.io/2019/04/25/recipe/) 
