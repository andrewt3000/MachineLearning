### Definition of Machine Learning
A program is said to learn from experience E with respect to some task T & some performance measure P, if it’s performance on T, as measured by P, improves with experience E.  
T - task  
P - probability of improved performance  
E - experience  

### 3 Types of machine learning
1. Supervised learning - learns uses labeled training data.  Training data has input and output.  
2. Unsupervised learning - learns using unlabled training data.  Training data is input but has no output.  Algorithm finds structure. Clustering is an example of unsupervised learning. Examples: social network analysis, market segmentation, astronomical data analysis.  
3. Reinforcement learning - learns to take action by maximizing a cumulative reward.

### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. Example: benign/malignant tumor, fraudulent or legitimate financial transaction, spam or non-spam email.

### 
[Linear Regression with one variable](https://github.com/andrewt3000/MachineLearning/blob/master/LinearRegression.md)  

[Neural Network Playground, simulator](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.28720&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false)  

### Terminology
Features - measurable property being observed. i.e. the input to a neural network.  
-example: pixel brightness in image object recognition.  
-example: square feet for program that predicts house prices.  

Feature selection - the process of choosing the features. It is critical, you must pick features that correlate with output.  
Feature scaling - scale features to be approximately in the range of -1 to 1.  

Hyperparameters - the model’s parameters in a neural net such as learning rate, and regularization factor.	

Learning rate (&alpha;) - controls the size of the adjustments made during the training process.  
if &alpha; is too low, convergance is slow.
if &alpha; is too high, there is no convergance, because it overshoots the local minimum.  

Regularization - variable to control overfitting.  

Underfitting - output doesn't fit the training data well. a.k.a has high bias.  
Overfitting - output fits training data well, but doesn't generalize well. aka has high variance.  

Architecture - The structure of the network i.e. number of hidden layers, and nodes. 

