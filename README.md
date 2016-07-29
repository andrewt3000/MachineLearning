[Neural Networks demystified video](https://www.youtube.com/watch?v=bxe2T-V8XRs)  

[TensorFlow Neural Network Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.28720&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false)  - This demo lets you run a neural network in your browser and see results graphically. Be sure to click the play button to start training. The network can easily train for the first three datasets with default parameters but the challenge is to get the network to train to the spiral dataset.  

[Linear Regression with one variable](https://github.com/andrewt3000/MachineLearning/blob/master/LinearRegression.md)  


### 3 Types of machine learning
1. Supervised learning - learns using labeled training data.  Training data has input and output.  
2. Unsupervised learning - learns using unlabled training data.  Training data is input but has no output.  Algorithm finds structure. Clustering is an example of unsupervised learning. Examples: social network analysis, market segmentation, astronomical data analysis.  
3. Reinforcement learning - learns to take action by maximizing a cumulative reward. [deep mind learns breakout](https://www.youtube.com/watch?v=V1eYniJ0Rnk)  

### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. Example: benign/malignant tumor, fraudulent or legitimate financial transaction, spam or non-spam email.

### Terminology
Features - measurable property being observed. i.e. the input to a neural network.  
Examples of features are pixel brightness in image object recognition, or square feet for program that predicts house prices.  
Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. For instance picking area of a house would be a more concise feature than using width and length.  
  
Feature scaling - scale features to be approximately in the range of -1 to 1.  

Hyperparameters - the model’s parameters in a neural net such as learning rate, and regularization factor.	

Learning rate (&alpha;) - controls the size of the adjustments made during the training process. A typical value is .1 but often the value is a smaller number.  
if &alpha; is too low, convergance is slow.
if &alpha; is too high, there is no convergance, because it overshoots the local minimum.  
The learning rate is often reduced to a smaller number over time. This is often called annealing or decay. (examples: step decay, exponential decay)  

Underfitting (high bias) - output doesn't fit the training data well.  
Overfitting (high variance) - output fits training data well, but doesn't work well on test data.  

Regularization (&lambda;) - variable to control overfitting. If model is underfitting, you need lower &lambda;. If the model is overfitting, you need higher lambda.

Dropout - a form of regularization.  

Architecture - The structure of the network i.e. number of hidden layers, and nodes. 

Number of hidden layers - the higher the number of layers the better it can find non-linear patterns, but the gradient also vanishes.  [Resnet](https://arxiv.org/abs/1512.03385) is on the high end at 152 layers.

[Practical tips for deep learning](http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html)

Epoch - number of times to iterate over the training data. An epoch means you have seen every training example once.

Mini batch size: Mini batches speed up the training process. Batch size determines how many training examples you consider before making a weight update. As the batch number gets higher it speeds up the process more, but becomes more noisey. Typical values are factors of 2, such as 32 or 128.



