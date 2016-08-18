### Neural Networks
Neural Network - an acyclical directed graph. Neural nets have input layers, hidden layers and output layers. Layers are connected by weighted synapsis that multiply their input times the weight. Hidden layer consists of neurons that sum their inputs from synapsis and execute an activation function on the sum. The weights are intially set to random values but are trained with backpropagation.  The input and output are of fixed size. Often called artificial neural networks, to distinguish it from biological, or feedforward neural network to distinguish from more complicated neural nets with feedback mechanisms. 

[thorough and concise resource on neural nets](http://frnsys.com/ai_notes/machine_learning/neural_nets.html)  

[Neural Networks demystified video](https://www.youtube.com/watch?v=bxe2T-V8XRs) - videos explaining neural networks. Includes [notes](https://github.com/stephencwelch/Neural-Networks-Demystified).    

[TensorFlow Neural Network Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.28720&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false)  - This demo lets you run a neural network in your browser and see results graphically. Be sure to click the play button to start training. The network can easily train for the first three datasets with default parameters but the challenge is to get the network to train to the spiral dataset.  

### Features / Input
Features - measurable property being observed. In neural net context, it's  the input to a neural network.  Examples of features are pixel brightness in image object recognition, words encoded as vectors in nlp applications, audio signal in voice recognition applications.  
  
Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. 

Dimensionality reduction - Reducing number of variables.  A simple example is selecting the area of a house as a feature rather than using width and length seperately. Another example is singular value decomposition.      

Feature scaling - scale each feature to be in a common range typically -1 to 1 where 0 is the mean value.    

### Hyperparameters
Hyperparameters - the model’s parameters in a neural net such as architecture, learning rate, and regularization factor.	

Architecture - The structure of a neural network i.e. number of hidden layers, and number of nodes. 

Number of hidden layers - the higher the number of layers the more layers of abstraction it can represent. too many layers and the the network suffers from the vanishing or exploding gradient problem.  

Learning rate (&alpha;) - controls the size of the adjustments made during the training process. A typical value is .1 but often the value is a smaller number.  
if &alpha; is too low, convergance is slow.
if &alpha; is too high, there is no convergance, because it overshoots the local minimum.  
The learning rate is often reduced to a smaller number over time. This is often called annealing or decay. (examples: step decay, exponential decay)  

Underfitting (high bias) - output doesn't fit the training data well.  
Overfitting (high variance) - output fits training data well, but doesn't work well on test data.  

Regularization - a technique to minimize overfitting. L1 and L2 are examples of regularization.  

[Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) - a form of regularization. "The key idea is to randomly drop units (along with their connections) from the neural network during training." Typical value is .5 (50%). As dropout value approaches zero dropout has less effect, as it approaches 1 there are more connections are being zeroed out. See [Hinton's dropout in 3 lines of python](https://iamtrask.github.io/2015/07/28/dropout/)      


###Activation Functions
Activation function - the "neuron" in the neural network executes an activation function on the sum of the weighted inputs. Typical activation functions include sigmoid, tanh, and ReLu.  

Sigmoid activation functions outputs a value between 0 and 1.  
```python
#sigmoid activation function using numpy
def sigmoid(z):
    return 1/(1+np.exp(-z))
```
Tanh activation function outputs value between -1 and 1.  

ReLu activation is typically the activation function used in state of the art convolutioanal neural nets for image classification.  

Softmax can be used as the output's activation function. Useful for modeling probability distributions. Output value is range [0, 1]. The sum of outputs is 1. Sums over all the output neurons in the denominator. Use with cross entropy cost function.  

###Training a network
Training data - Input and labeled output used as training examples. Data is typically split into training data, cross validation data and test data. Typical mix is 60% training, 20% validation and 20% testing data. Validation is used to tune the model and it's hyperparameters. Testing uses data that the model was never trained on.  

Training a network - minimize a cost function. Use backpropagation and gradient descent to adjust weights to make model more accurate. 

Cost Function, aka error function or loss function - measures how inaccurate a model is. Training a model minimizes the cost function. Sum of squared errors is a common cost function for regression. Cross entropy is a common cost function for multi-class classification and softmax activations.  

Backpropagation - computes the gradient of the loss function with respect to the weights in the network. Uses a non-linear optimization method such as stochastic gradient descent, momentum sgd, nesterov momentum sgd, adagrad, and adaDelta. 

Gradient descent - Iteratively adjusting the weight by learning rate times the gradient.  
There is batch gradient descent and stochastic gradient descent. Batch uses entire dataset and works well for convex errors. Stochastic gradient descent uses a single example and works better if there are lots of minima and maxima. sgd often uses mini-batches. sgd is used more often in practice for deep learning. 

Number of times to iterate over the training data - Typically you run the program until the training there is no improvement for a long period. Hopefully the training and test losses are converging on an acceptablely low error level. An epoch means the network has been been trained on every example once.  

Mini batch size: Mini batches speed up the training process. Batch size determines how many training examples you consider before making a weight update. As the batch number gets higher it speeds up the process more, but becomes more noisey. Typical values are factors of 2, such as 32 or 128.

[Practical tips for deep learning](http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html)  

### Other Types of Neural Networks
Convolutional Neural Networks - Specialized to process a grid of information such as an image. Convolution neural networks use filters (aka kernels) that convolve over the grid.    

Recurrent Neural Network (RNN) - Used for input sequences such as text, audio, or video. RNNs are the same as a neural network but they also pass the hidden state output of each neuron via a weighted connection as an input to the neurons in the same layer during the next sequence. This feedback architecture allows the network to have memory of previous inputs. The memory is limited by vanishing/exploding gradient problem. RNNs are trained by backpropagation through time.  There are variations such as bi-directional and recursive RNNs.    

LSTM - [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) - A specialized RNN that is capable of long term dependencies. Contains memory cells and gates. The number of memory cells is a hyperparameter. Memory cells pass memory information forward. The gates decide what information is stored in the memory cells. A vanilla LSTM has a forget gates, input gates and output gates. There are [many variations of the LSTM](http://arxiv.org/pdf/1503.04069.pdf).  

GRU - Gated Recurrent Unit - Introduced by Cho. Another RNN variant similar but simpler than LSTM. It contains one update gate and combines the hidden state and memory cells among other differences.  

