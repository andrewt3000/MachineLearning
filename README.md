### My Notes on Machine Learning

Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

#### 3 Types of machine learning
1. Supervised learning - learns using labeled training data.  Training data has features and labels.  
2. Unsupervised learning - learns using unlabled training data.  Training data has features but no labels.  Algorithm finds structure. Clustering is an example of unsupervised learning. 
3. Reinforcement learning - an agent interacts with an environment and learns to take action by maximizing a cumulative reward. [David Silver RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)  

#### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)

### Dimensionality reduction 
Reducing number of features. A simple example is selecting the area of a house as a feature rather than using width and length seperately. Other examples include singular value decomposition, variational auto-encoders, and t-SNE (for visualizations), and max pooling layers for CNNs.

### Machine learning models and applications

[Neural Nets](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md) - A primer on neural networks.  NNs are a suitable model for fixed input features and output labels.    

[Recurrent Neural Nets](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md) - A primer on recurrent neural networks. RNNs are a suitable model for sequences of information.   

[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) CNNs basics. CNNS are suitable models for 2d grids of information such as an image. SOA for object classification: [res net](https://arxiv.org/abs/1512.03385), [Inception v4](https://arxiv.org/abs/1602.07261), [dense net](https://arxiv.org/abs/1608.06993)   

[Deep Learning for NLP](https://github.com/andrewt3000/DL4NLP/blob/master/README.md) State of the art deep learning models and nlp applications such as sentiment analysis, translation and dialog generation.  


### Advanced Topics  
Generative adversarial networks - GANs - 2 Neural networks compete against each other. The generative network generates candidates while the discriminative network evaluates them. This technique can generate realistic images. See [GAN](https://arxiv.org/abs/1406.2661), [Lap GAN](https://arxiv.org/abs/1506.05751), [DC GAN](https://arxiv.org/abs/1511.06434), [Big GAN](https://arxiv.org/abs/1809.11096), [StyleGAN](https://arxiv.org/abs/1812.04948)   

[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) - storing knowledge gained while solving one problem and applying it to a different but related problem.

[Neural architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search) - (a sub-field of automl) automatically designing neural networks architecture. [NAS Survey](https://arxiv.org/abs/1808.05377)  / [AutoML papers](https://www.automl.org/automl/literature-on-neural-architecture-search/) / Examples: [NAS](https://arxiv.org/abs/1611.01578), [ENAS](https://arxiv.org/abs/1802.03268), [PNAS](https://arxiv.org/abs/1712.00559), [DARTS](https://arxiv.org/abs/1806.09055)  

Feature visualization - In computer vision, generating images representative of what neural networks are looking for. (Example [Activation Atlas](https://distill.pub/2019/activation-atlas/) )  

Feature attribution - In computer vision, determining and representing which pixels contribute to a classification.   

