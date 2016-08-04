### 3 Types of machine learning
1. Supervised learning - learns using labeled training data.  Training data has input and output.  
2. Unsupervised learning - learns using unlabled training data.  Training data is input but has no output.  Algorithm finds structure. Clustering is an example of unsupervised learning. Examples: social network analysis, market segmentation, astronomical data analysis.  
3. Reinforcement learning - learns to take action by maximizing a cumulative reward. [deep mind learns breakout](https://www.youtube.com/watch?v=V1eYniJ0Rnk)  

### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. (Example: pass or fail)

#### Types of classification  
Binary classification - classifying elements into one of two groups. Example: benign/malignant tumor, fraudulent or legitimate financial transaction, spam or non-spam email.  

Softmax regression (aka multinomial logistic regression) - outputs multiple binary labels. Example: handwritten single numeric character classification has 10 binary outputs 0 - 9. the outputs are mutually exclusive and output should sum up to 1 predicting probability of each label.  

### Neural Network Basics
[Neural Networks demystified video](https://www.youtube.com/watch?v=bxe2T-V8XRs) - these videos explain neural networks to do a regression problem.  

[TensorFlow Neural Network Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.28720&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false)  - This demo lets you run a neural network in your browser and see results graphically. Be sure to click the play button to start training. The network can easily train for the first three datasets with default parameters but the challenge is to get the network to train to the spiral dataset.  

[Linear Regression with one variable](https://github.com/andrewt3000/MachineLearning/blob/master/LinearRegression.md)  

[Practical tips for deep learning](http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html)  

### Terminology
Features - measurable property being observed. In neural net context, it's  the input to a neural network.  Examples of features are pixel brightness in image object recognition, words encoded as vectors in nlp applications, audio signal in voice recognition applications, or square feet for program that predicts house prices.  
  
Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. 

Dimensionality reduction - Reducing number of variables.  A simple example is selecting the area of a house as a feature rather than using width and length seperately. Another example is singular value decomposition.      

Feature scaling - scale each feature to be in a common range typically -1 to 1 where 0 is the mean value.    

Hyperparameters - the model’s parameters in a neural net such as learning rate, and regularization factor.	

Learning rate (&alpha;) - controls the size of the adjustments made during the training process. A typical value is .1 but often the value is a smaller number.  
if &alpha; is too low, convergance is slow.
if &alpha; is too high, there is no convergance, because it overshoots the local minimum.  
The learning rate is often reduced to a smaller number over time. This is often called annealing or decay. (examples: step decay, exponential decay)  

Underfitting (high bias) - output doesn't fit the training data well.  
Overfitting (high variance) - output fits training data well, but doesn't work well on test data.  

Regularization - a technique to minimize overfitting. L1 and L2 are examples of regularization.  

[Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) - a form of regularization. "The key idea is to randomly drop units (along with their connections) from the neural network during training." Typical value is .5 (50%). As dropout value approaches zero dropout has less effect, as it approaches 1 there are more connections are being zeroed out. See [Hinton's dropout in 3 lines of python](https://iamtrask.github.io/2015/07/28/dropout/)      

Architecture - The structure of a neural network i.e. number of hidden layers, and number of nodes. 

Number of hidden layers - the higher the number of layers the better it can find non-linear patterns, but the gradient also vanishes or explodes.  [Resnet](https://arxiv.org/abs/1512.03385) is on the high end at 152 layers.

Activation function - the "neuron" in the neural network executes an activation function on the sum of the weighted inputs. Typical values include sigmoid, ReLu, and tanh.  

```python
#sigmoid activation function using numpy
def sigmoid(z):
    return 1/(1+np.exp(-z))
```

Cost Function, aka error function or loss function - measures how inaccurate a model is. training a model minimizes the cost function. Sum of squared errors is a common cost function for regression. Cross entropy (aka log loss or logistic loss) is a common cost function for binary classification.  

Number of times to iterate over the training - Typically you run the program until the training there is no improvement for a long period. Hopefully the training and test losses are converging on an acceptablely low error level. An epoch means the network has been been trained on every example once.  

Mini batch size: Mini batches speed up the training process. Batch size determines how many training examples you consider before making a weight update. As the batch number gets higher it speeds up the process more, but becomes more noisey. Typical values are factors of 2, such as 32 or 128.

Batch gradient descent versus stochastic gradient descent: batch uses entire dataset and works well for convex errors. stochastic gradient descent uses a single example and works better if there are lots of minima and maxima. sgd often uses mini-batches. sgd is used more often in practice.  

### Types of Neural Networks
Neural Network - Has input layers, hidden layers and output layers. Hidden layer consists of neurons that execute an activation function. Hidden layers are connected by synapsis that are weighted. The weights are intially set to random values but are trained with backpropagation.  The input and output are of fixed size. Often called artificial neural networks, to distinguish it from biological, or feedforward neural network. 

Convolutional Neural Networks - Specialized to process a grid of information such as an image. Convolution neural networks use filters (aka kernels) that convolve over the grid.    

Recurrent Neural Network (RNN) - Used for input sequences such as text, audio, or video. RNNs are the same as a neural network but they also pass the hidden state output of each neuron via a weighted connection as an input to the neurons in the same layer during the next sequence. This feedback architecture allows the network to have memory of previous inputs. The memory is limited by vanishing/exploding gradient problem. RNNs are trained by backpropagation through time.  There are variations such as bi-directional and recursive RNNs.    

LSTM - [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) - A specialized RNN that is capable of long term dependencies. Contains memory cells and gates. The number of memory cells is a hyperparameter. Memory cells pass memory information forward. The gates decide what information is stored in the memory cells. A vanilla LSTM has a forget gates, input gates and output gates. There are [many variations of the LSTM](http://arxiv.org/pdf/1503.04069.pdf).  

GRU - Gated Recurrent Unit - Introduced by Cho. Another RNN variant similar but simpler than LSTM. It contains one update gate and combines the hidden state and memory cells among other differences.  

### Applications

[Convolutional Neural Networks for image recognition](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md)  

[Deep Learning for NLP](https://github.com/andrewt3000/DL4NLP/blob/master/README.md)
