# Calculus for Machine Learning

Training a neural network is an optimization problem: find the parameters that minimize a loss function. Calculus provides the tools — derivatives tell you which direction to adjust each parameter, and the chain rule ([backpropagation](neuralNets.md#backpropagation)) computes those adjustments efficiently through deep networks. This page covers the small subset of calculus that machine learning actually uses.

### Derivatives
The **derivative** of a function measures its instantaneous rate of change — the slope of the function at a point. Notation: f'(x) or df/dx.

- If the derivative is positive, the function is increasing at that point.
- If the derivative is negative, the function is decreasing.
- If the derivative is 0, the point is a **critical point**: a minimum, maximum, or saddle point.

In machine learning, the function is the loss and the inputs are the weights. The derivative of the loss with respect to a weight tells you: if I increase this weight slightly, does the loss go up or down, and by how much?

#### Common derivatives used in ML
| Function | Derivative | Where it appears |
| :--- | :--- | :--- |
| x² | 2x | mean squared error |
| eˣ | eˣ | softmax, sigmoid |
| ln(x) | 1/x | cross entropy / log loss |
| c (constant) | 0 | bias terms, dead ReLUs |
| cx | c | linear layers |

### Partial derivatives and the gradient
A neural network's loss is a function of millions of weights, not a single variable. A **partial derivative** (∂J/∂w) is the derivative with respect to one variable while holding all the others constant.

The **gradient** (∇J) is the vector of all the partial derivatives — one entry per parameter. The gradient points in the direction of **steepest ascent** of the loss. Gradient descent therefore steps in the *negative* gradient direction:

w := w − α ∇J(w)

where α is the [learning rate](neuralNets.md#learning-rate). This single update rule, applied repeatedly, is how neural networks train.

### The chain rule
The **chain rule** computes the derivative of composed functions:

if y = f(g(x)), then dy/dx = f'(g(x)) · g'(x)

A neural network is a deep composition of functions — layer after layer of linear transformations and activations, ending in a loss:

J = Loss(Softmax(W₂ · ReLU(W₁ · X)))

To get the gradient of the loss with respect to an early weight, the chain rule multiplies the local derivatives of every function between that weight and the loss. **Backpropagation** is the chain rule applied recursively from the output backwards, reusing intermediate results so each layer's gradient is computed once.

The chain rule also explains **vanishing and exploding gradients**: the gradient at layer 1 of a deep network is a product of many terms. If those terms are mostly less than 1, the product shrinks toward zero (vanishing); if mostly greater than 1, it blows up (exploding). This is why [initialization](neuralNets.md#initialization), ReLU activations, and residual connections matter for deep networks.

### Convexity
A function is **convex** if it curves upward everywhere — any local minimum is the global minimum. A bowl is convex; an egg carton is not.

- Linear regression with squared error and logistic regression with cross entropy have **convex** loss functions, so gradient descent is guaranteed to find the global minimum.
- Neural network losses are **non-convex**: the loss surface has many critical points. In high dimensions, saddle points are far more common than bad local minima, and the noise in [stochastic gradient descent](neuralNets.md#optimization-algorithms) helps escape them.

### Second derivatives
The **second derivative** measures curvature — how fast the slope itself is changing. Its multivariable generalization is the **Hessian** matrix of all second partial derivatives.

- Second-order optimizers (Newton's method) use curvature to take smarter steps but are impractical for deep learning: the Hessian of a model with n parameters has n² entries.
- Optimizers like [Adam](neuralNets.md#gradient-descent-optimization) approximate the benefit cheaply by adapting the learning rate per parameter using running estimates of gradient statistics.
- "Sharp" versus "flat" minima (curvature of the loss around a solution) is the standard language for discussing why large-batch training can generalize worse.

### Integrals
Integration (area under a curve) appears less in daily deep learning practice, but shows up in:
- **Probability**: continuous distributions integrate to 1; expected values are integrals. The "E" in expressions like E[X] denotes an expectation.
- **Expected loss**: training minimizes the average loss over data, which approximates an expectation (an integral over the data distribution).
- **ROC-AUC**: the area under the ROC curve is literally an integral.

### Automatic differentiation
In practice you rarely derive gradients by hand. **Automatic differentiation** (autograd) records the operations in the forward pass as a computation graph, then applies the chain rule mechanically in reverse. This is exact (not a numerical approximation) and is what PyTorch's [backward()](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) does.

**Gradient checking** — comparing autograd's result against a numerical approximation (f(x+ε) − f(x−ε)) / 2ε — is a debugging technique for verifying hand-written gradients.

### References / tutorials
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - visual intuition for derivatives and the chain rule
- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - the chain rule applied to a neural network, step by step
- 2018 Parr & Howard [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) - bridges this page and matrix/vector calculus
- CS231n [notes on backpropagation](https://cs231n.github.io/optimization-2/) - computation graphs and gradient flow
