###Linear Regression with one variable
m = number of training examples  
x = input features  
y = output variable / target variable  

(x, y) = refers to one training example  
x<sup>(i)</sup> y<sup>(i)</sup> refers to specific training example at index i. It doesn’t refer to an exponent.

h = hypothesis. function that maps x to y  
Predict that y is a linear function of x.  
<strong>y = h(x) = ϴ<sub>0</sub> + ϴ<sub>1</sub>x</strong>  

###Cost Function
J is the cost function  
ϴ<sub>i</sub> are parameters. the coefficients of the function.  
find values of ϴ<sub>0</sub> and ϴ<sub>1</sub> that minimize the cost (error) function.  
h(x) - y = difference in function versus actual… we want to minimize this.  
aka squared error cost function.  
J(ϴ<sub>0</sub>, ϴ<sub>1</sub>) = 1/(2m) &#931; from i = 1 to m (h(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup>  
find ϴ<sub>0</sub> and ϴ<sub>1</sub> that minimizes the error.  

Square the error because  
1) Squared gets rid of negative numbers that would cancel each other out.  Although you could use magnitude.  
2) For many applications small errors are not important, but big errors are very important.  example: self driving car. ½ foot steering error no big deal, 5 foot error fatal problem. So it’s not 10x more important… it is 100x more important.  
3) The convex nature of quadratic equation avoids local minimums.  

###Gradient Descent
:=  (assignment operator, not equality)  
&alpha; = learning rate.  the learning rate controls the size of the adjustments made during the training process. 

Repeat until convergence.  
&theta;<sub>0</sub> := &theta;<sub>0</sub> - &alpha; (1/m) &#931; i = 1 to m (h<sub>&theta;</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)  
&theta;<sub>1</sub> := &theta;<sub>1</sub> - &alpha; (1/m) &#931; i = 1 to m (h<sub>&theta;</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>) &bull; x<sup>(i)</sup>  

### Logistic Regression
Regularization factor (&lambda;) - variable to control overfitting. If model is underfitting, you need lower &lambda;. If the model is overfitting, you need higher lambda.

