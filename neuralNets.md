### Neural Networks
[Neural Network](https://en.wikipedia.org/wiki/Neural_network) - Nueral networks are machine learning models and [universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem).  
Neural nets have input layers, hidden layers and output layers.  
Layers are connected by weighted synapsis (the lines with arrows) that multiply their input times the weight. 
Hidden layer consists of neurons (the circles) that sum their inputs from synapsis and execute an activation function on the sum.  
The layers are of fixed size. Neural networks also typically have a single bias input node that is a constant value. It's similar to the constant in a linear function.  

Aka artificial neural networks, feedforward neural networks, vanilla neural networks.   

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/nn.png" height='250px' width='250px'/>  

### Features / Input
[Features](https://en.wikipedia.org/wiki/Feature_(machine_learning)) - measurable property being observed. In neural net context, it's the input layer to a neural network.  Examples of features are pixel brightness in image object recognition, words encoded as vectors in nlp applications, audio signal in voice recognition applications.  
  
Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. 

### Training Data 
Training data with a larger number of diverse examples reduces likelihood of overfitting.  

It is important to have a balanced number of examples for each label. You're likely to see better results if you maintain at least a 1:2 ratio between the label with the fewest examples and the label with the most examples. 

[Microsoft CV guidelines](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/getting-started-improving-your-classifier)

### Hyperparameters
Hyperparameters - the model’s parameters in a neural net such as architecture, learning rate, and regularization factor.	

Architecture - The structure of a neural network i.e. number of hidden layers, and number of nodes. 

Number of hidden layers - the higher the number of layers the more layers of abstraction it can represent. too many layers and the the network suffers from the vanishing or exploding gradient problem.  

### Activation Functions
[Activation function](https://en.wikipedia.org/wiki/Activation_function) - the "neuron" in the neural network executes an activation function on the sum of the weighted inputs. In the neuron metaphor you can assume as the value approaches 1 the neuron is "firing". ReLu is a popular modern activation function.  

#### Sigmoid
Sigmoid activation functions outputs a value between 0 and 1. It is a smoothed out step function. Sigmoid is not zero centered and it suffers from activation saturation issues. Historically popular, but not currently popular.  

#### Tanh
Tanh activation function outputs value between -1 and 1. Tanh is a rescaled sigmoid function. Tanh is zero centered but still suffers from activation saturation issues similar to sigmoid. Historically popular, but not currently popular.  

#### ReLu
ReLu activation is currently (2018) the most popular activation function. ReLu stands for rectified linear unit. It returns 0 for negative values, and the same number for positive values. Relu can suffer from "dead" relus (vanishing gradient?)    

```python
def relu(x):
  if x < 0:
    return 0
  if x >= 0:
    return x
```

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/activation.png" />

#### More activations
There are newer, experimental variants of the relu: Leaky ReLu (solves the dead relu issue) Elu (exponential relu), and MaxOut.   

#### Softmax
The [softmax function](https://en.wikipedia.org/wiki/Softmax_function) is often used as the model's final output activation function for classification. Softmax is used for modeling probability distributions for classification where outputs are mutually exclusive (MNIST is an example). 
Softmax is a "soft" maximum function. It's properties are:  
Output values are in the range [0, 1].  
The sum of output nodes is 1.  

The softmax function as applied to each node NN output is the exponent of the output divided by the sum of all the exponent outputs. So for instance, if there are 3 nodes, the output of the 1st node y1 is:     
e <sup>y^1</sup> / (e <sup>y^1</sup> + e <sup>y^2</sup> + e <sup>y^3</sup>)

```python
def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)
```

See additonal activations in [Keras activations](https://keras.io/activations/)  

### Training a neural network
Training data - The data used in machine learning models to make the model more accurate. In the case of supervised learning it consists of input features and output labels that serve as examples. Data is typically split into training data, cross validation data and test data. Typical mix is 60% -80% training, 10%-20% validation and 10%-20% testing data. Validation data is evaluated while the model is training and it indicates if the model is generalizing. Testing data is evaluated after you have stopped training the model to indicate the model's accuracy.  

Training a neural network - minimize a cost function. Use backpropagation and gradient descent to adjust weights to make model more accurate. 

Steps to training a network.  
- Prepare the data
- initialize weights and biases.  
- Implement forward propagation  
- Implement cost (aka error, loss) function
- Implement backpropagation 
- Run optimization algorithm such as gradient descent adjusting weights using learning rate.  

You train until the model (hopefully) converges on an acceptablely low error level. An epoch means the network has been trained on every example once.  

One tip is to begin by overtraining a small portion of your data by getting rid of regularization to see if you have a reasonalbe architecture. If you can't get the loss to zero, reconsider your architecture.   

Start with small regularization and find a learning rate that makes the loss go down.  

### Prepare the data
[Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) - scale each feature to be in a common range typically -1 to 1 where 0 is the mean value. For instance, in image processing you could subtract the mean pixel intensity for the whole image or by channel before scaling it to zero center the features.    

[Data augmentation](https://en.wikipedia.org/wiki/Data_augmentation) - An example of data augmentation for images is mirroring, flipping, rotating, and translating your images to create new examples.   

### Initialization
The weights and biases are historically initialized with small random numbers centered on zero. If the weights are the same (say all 0s) they will remain the same throughout training, making the weights random breaks this symmetry.  

As your neural networks get deeper, initialization becomes more important. If the initial weights are too small, you get a vanishing gradient. If the initial weights are too large, you get an exploding gradient. Consider more advanced initializations such as Xavier and He initialization.   

In Keras, you can specify kernel and bias initialization on each Dense layer. See all available [keras initializations](https://keras.io/initializers/). Glorot (aka Xavier) initialization is the default.  

### Forward Propagation
If X is the input matrix, and W1 is the weight matrix for the first hidden layer, we take the dot product to get the values passed to the activation functions. Then we apply the activation function to each element in the matrix. Repeat for each layer.  

```python
z2 = np.dot(X, W1)
a2 = activation(z2)
```

[Example of Forward propagation in numpy](https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/.ipynb_checkpoints/Part%202%20Forward%20Propagation-checkpoint.ipynb)

### Loss Function 
Loss (aka cost/error/objective) function measures how inaccurate a model is. Training a model minimizes the loss function. Sum of squared errors is a common loss function for regression. Cross entropy (aka log loss, negative log probability) is a common loss function for softmax function.   

Cross entropy function is suitable for a classification where the output is a value between 0 and 1. The loss will be 0 if the output value is 1 and that is the correct classification. Conversely, the loss approaches infinity as the output approaches 0 for the correct classification.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/cross_entropy.png" />

```python
def logloss(true_label, predicted_prob):
  if true_label == 1:
    return -log(predicted_prob)
  else:
    return -log(1 - predicted_prob)
 ```

[reference](http://wiki.fast.ai/index.php/Log_Loss)  
[Keras loss functions](https://keras.io/losses/)  

### Backpropagation 
The backpropagation algorithm applies the chain rule recursively to compute the gradient for each weight. The gradient is caulated by taking the partial derivative of the loss function with respect to the weights at each layer of the network by moving backwards (output to input) through the network. Backprop indicates how to adjust the weights to minimize the loss function. If the gradient (i.e. partial derivative/slope) is positive, that means the loss is getting higher as the weight increases. If the derivative is 0, the weight is set to a minimum loss. The gradient indicates the magnitude and direction of adjustments to our weights that will reduce the loss.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/descent.png"   height='360px' width='640px' />

We can combine all these functions for cost and forward propagation to get one function. So for instance, the cost function for a NN with one hidden relu layer using a softmax output is (where W1 and W2 are weights for 1st and 2nd layers, and X1 is the input feature nodes:  
  
```
J = Cost(Softmax(DotProduct(Relu(DotProduct(X1,W1)), W2)))
```

We then calculate the partial derivative of the loss function with respect to each weights (dJ/dW). We use the [chain rule](https://en.wikipedia.org/wiki/Chain_rule). The chain rule is the derivative of f(g(x)) is f'(g(x)) * g'(x)

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

Packages such Keras handle this part of the training process for you.  

#### Learning Rate
Learning rate (&alpha;) - controls the size of the adjustments made during the training process. Typical values are .1, .01, .001. Consider these values are relative to your input features which are typically scaled to ranges such as 0 to 1, or -1 to +1.  
if &alpha; is too low, convergance is slow.
if &alpha; is too high, there is no convergance, because it overshoots the local minimum.  
The learning rate is often reduced to a smaller number over time. This is often called annealing or decay. (examples: step decay, exponential decay)  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/lr.jpg" />

### Optimization algorithms
[Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is an iterative optimization algorithm that, in the context of neural networks, adjusts the weight by learning rate times the negative of the gradient (calculated by backpropagation) to mimimize the loss function.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/gd.jpg"  height='360px' width='640px' />

Batch gradient descent - The term batch refers to the fact it uses the entire dataset to make one gradient step. Batch works well for small datasets that have convex loss functions. The loss function needs to be convex or it may find a local minimum.    

[Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (sgd) is a variation of gradient descent that uses a single randomly choosen example to make an update to the weights. sgd is more scalable than batch graident descent and is used more often in practice for large scale deep learning. It's random nature makes it unlikely to get stuck in a local minima.  

Mini batch gradient descent: Stochastic gradient descent that considers more than one randomly choosen example before making an update. Batch size is a hyperparmeter that determines how many training examples you consider before making a weight update. Typical values are factors of 2, such as 32 or 128. Values are typically in the range of 32-512.  Larger batches are faster to train, but can cause overfitting and require more memory.  Lower batch sizes are the opposite: slower to train, more regularized, and require less memory.  

#### Gradient Descent Optimization
Momentum sgd is a variation that accelerates sgd, dampens oscillations, and helps skip over local minima and saddlepoints. It collects data on each update in a velocity vector to assist in calculating the gradient. The velocity matrix represents the momentum. Rho is a hyperparameter that represents the friction. Rho is in the range of 0 to 1. Typical values for rho are 0.9 and 0.99. 

Other optimization algorithms include: Nesterov, AdaGrad, AdaDelta, Adam, Adamax, NAdam, RMSProp, and AMSGrad.  
See [10 Gradient Descent Optimisation Algorithms in a Cheat Sheet](https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9)  
See [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)  
See [Keras Optimizers](https://keras.io/optimizers/)  

#### Regularization
Underfitting - output doesn't fit the training data well.  
[Overfitting](https://en.wikipedia.org/wiki/Overfitting) - output fits training data well, but doesn't work well on validation or test data.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/over_under.png"/>

Regularization - a technique to minimize overfitting.  

L1 regularization uses sum of absolute value of weights. L1 works best with sparse outputs.  
L2 regularization uses sum of squared weights. L2 doesn't work well with yielding sparse outputs.    

[Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) - a form of regularization. "The key idea is to randomly drop units (along with their connections) from the neural network during training." Typical hyperparameter value is .5 (50%). As dropout value approaches zero, dropout has less effect, as it approaches 1 there are more connections are being zeroed out. The remaining active connections are scaled up to compensate for the zeroed out connections. Dropout is typically implemented in training but not present in inference. Dropout is typically applied to units in the hidden layers.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/dropout.png" />  

Early termination - Stop training when the training error is getting lower but the validation error is increasing. This indicates overfitting.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/early_term.png" />


### Other resources
[Neural Networks demystified video](https://www.youtube.com/watch?v=bxe2T-V8XRs) - videos explaining neural networks. Includes [notes](https://github.com/stephencwelch/Neural-Networks-Demystified).    

[Stanford CS231n](http://cs231n.github.io/neural-networks-1/)  

[TensorFlow Neural Network Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.28720&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false)  - This demo lets you run a neural network in your browser and see results graphically. I wrote about the lessons on [intuition about deep learning]([https://medium.com/@andrewt3000/understanding-tensorflow-playground-c20cdb7a250b]).   

[Practical tips for deep learning.  Ilya Sutskever](http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html)  

[A Recipe for Training Neural Networks - Andrej Karpathy](http://karpathy.github.io/2019/04/25/recipe/) 
