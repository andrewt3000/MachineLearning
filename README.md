### My Notes on Machine Learning

Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

#### 3 Types of machine learning
1. Supervised learning - learns using labeled training data.  Training data has features and labels.  
2. Unsupervised learning - learns using unlabled training data.  Training data has features but no labels.  Algorithm finds structure. Clustering is an example of unsupervised learning. 
3. Reinforcement learning - an agent interacts with an environment and learns to take action by maximizing a cumulative reward. [David Silver RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)  

#### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)

### Machine learning models and applications

[Neural Nets](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md) - A primer on neural networks.  NNs are a suitable model for fixed input features and output labels.    

[Recurrent Neural Nets](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md) - A primer on recurrent neural networks. RNNs are a suitable model for sequences of information.   

[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) A primer for CNNs. CNNS are suitable models for 2d grids of information such as an image.   

[Deep Learning for NLP](https://github.com/andrewt3000/DL4NLP/blob/master/README.md) State of the art deep learning models and nlp applications such as sentiment analysis, translation and dialog generation.  


### Advanced Topics  

[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) - storing knowledge gained while solving one problem and applying it to a different but related problem.

[Neural architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search) - (a sub-field of automl) automatically designing neural networks architecture. [NAS Survey](https://arxiv.org/abs/1808.05377)  / [AutoML papers](https://www.automl.org/automl/literature-on-neural-architecture-search/) / Examples: [NAS](https://arxiv.org/abs/1611.01578), [ENAS](https://arxiv.org/abs/1802.03268), [PNAS](https://arxiv.org/abs/1712.00559), [DARTS](https://arxiv.org/abs/1806.09055)  

Feature visualization - In computer vision, generating images representative of what neural networks are looking for. (Example [Activation Atlas](https://distill.pub/2019/activation-atlas/) )  

Feature attribution - In computer vision, determining and representing which pixels contribute to a classification.   

Generative adversarial networks - GANs - 2 Neural networks constest each other. The generative network generates candidates while the discriminative network evaluates them. This technique can generate realistic images. See [StyleGAN](https://arxiv.org/abs/1812.04948)   
