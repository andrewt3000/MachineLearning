Notes for [Coursera: Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome?module=tN10A)  

#### 3 Types of machine learning
1. Supervised learning - learns using labeled training data.  Training data has input and output.  
2. Unsupervised learning - learns using unlabled training data.  Training data is input but has no output.  Algorithm finds structure. Clustering is an example of unsupervised learning. Examples: social network analysis, market segmentation, astronomical data analysis.  
3. Reinforcement learning - learns to take action by maximizing a cumulative reward. [deep mind learns breakout](https://www.youtube.com/watch?v=V1eYniJ0Rnk)  

#### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. (Example: pass or fail)

#### Types of classification  
Binary classification - classifying elements into one of two groups. Examples: benign/malignant tumor, fraudulent or legitimate financial transaction, spam or non-spam email.  

[Multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification)/multinomial classification - classify instances into more than 2 classes. Example: MNIST evaluates handwritten single numeric characters and classifies into 10 binary classes 0 - 9.  

###Week 1: Linear Regression with one variable

m = number of training examples  
x = input features  
y = output variable / target variable  

(x, y) = refers to one training example  
x<sup>(i)</sup> y<sup>(i)</sup> refers to specific training example at index i. It doesn’t refer to an exponent.

h = hypothesis. function that maps x to y  
Predict that y is a linear function of x.  
<strong>y = h(x) = ϴ<sub>0</sub> + ϴ<sub>1</sub>x</strong>  

####Cost Function
J is the cost function  
ϴ<sub>i</sub> are parameters. the coefficients of the function. Similar to weights in a neural net.  
find values of ϴ<sub>0</sub> and ϴ<sub>1</sub> that minimize the cost (error) function.  
h(x) - y = difference in function versus actual… we want to minimize this.  
aka squared error cost function.  
J(ϴ<sub>0</sub>, ϴ<sub>1</sub>) = 1/(2m) &#931; from i = 1 to m (h(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup>  
find ϴ<sub>0</sub> and ϴ<sub>1</sub> that minimizes the error.  

Square the error because  
1) Squared gets rid of negative numbers that would cancel each other out.  Although you could use magnitude.  
2) For many applications small errors are not important, but big errors are very important.  example: self driving car. ½ foot steering error no big deal, 5 foot error fatal problem. So it’s not 10x more important… it is 100x more important.  
3) The convex nature of quadratic equation avoids local minimums.  

####Gradient Descent
:=  (assignment operator, not equality)  
&alpha; = learning rate.  the learning rate controls the size of the adjustments made during the training process. 

Repeat until convergence.  
&theta;<sub>0</sub> := &theta;<sub>0</sub> - &alpha; (1/m) &#931; i = 1 to m (h<sub>&theta;</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)  
&theta;<sub>1</sub> := &theta;<sub>1</sub> - &alpha; (1/m) &#931; i = 1 to m (h<sub>&theta;</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>) &bull; x<sup>(i)</sup>  

###Week 2 MultiVariate Linear Regression
n = the number of features  
Feature scaling - get features in the range -1 to 1.  
x<sub>i</sub> = x<sub>i</sub> - average / range (max - min)  

###Week 3 Logistic regression
Logistic regression is a confusing term because it is a classification algorithm, not a regression algorithm.  
"Logistic regression" is named after the logistic function (aka sigmoid function)  
The sigmoid function is the hypothesis for logistic regression: h(x)= 1 / (1 + e<sup>-&theta; T x</sup>)  
The cost function for logistic regression is:  
for y = 1, -log(h<sub>&theta;</sub>(x))  
for y = 0, -log(1 - h<sub>&theta;</sub>(x))   

#####Multiclass classification
Multiclass classification is solved using "One versus All." There are K output classes. Solve for K binary logistic regression classifiers and choose the one with the highest probability.    

####Regularization
Underfitting (high bias) - output doesn't fit the training data well.  
Overfitting (high variance) - output fits training data well, but doesn't work well on test data. Failes to generalize.  

Regularization factor (&lambda;) - variable to control overfitting. If model is underfitting, you need lower &lambda;. If the model is overfitting, you need higher lambda.  

The intuition of &lambda; is that you add the regularization term to the cost function, and as you minimize the cost function, you minimize the magnitude of theta (the weights). If theta is smaller, especially for higher order polynomials, the hypothesis is simpler.  

The regularization term is added to the cost function.  
The regularization term is for linear regression is:  &lambda; times the sum from j=1 to n of &theta;<sub>j</sub><sup>2</sup>  
The regularization term is for logistic regression is:  &lambda;/2m times the sum from j=1 to n of &theta;<sub>j</sub><sup>2</sup>  
Notice they sum from j=1, not j=0. i.e. it doesn't consider the bias term.  

###Week 5 Neural Networks
L is the number of layers.  
K number of output units.  


####How to train a neural network:
1. randomize initial weights.
2. Implement forward propagation. hϴ(xi)
3. Implement cost function J(ϴ)
4. Implement backprop to compute partial derivative of cost function with respect to theta.
 for 1 through m  (each training example)
5. Do gradient checking
6. Use gradient descent.

### Week 8 Unsupervised learning and dimensionality reduction
K-means clustering. Example using 2 clusters. Randomly initialize 2 cluster centroids. K-means is an iterative algorithm. 
Step 1 is to assign each data point to the nearest cluster centroid. Step 2: move the cluster centroid to the location of the mean of the assigned data points.  

K = number of clusters to find.  
