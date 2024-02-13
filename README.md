### My Notes on Machine Learning

Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

#### 3 Types of machine learning
1. Supervised learning - Minimize an error function using labeled training data.  
2. Unsupervised learning - Find patterns using unlabled training data. Examples include Principal component analysis and clustering. 
3. Reinforcement learning - Maximize a reward. An agent interacts with an environment and learns to take action by maximizing a cumulative reward.   

### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)

### ML Terminology
[Features](https://en.wikipedia.org/wiki/Feature_(machine_learning)) - measurable property being observed. Features are the inputs to a machine learning model. An example of a features is pixel brightness in computer vision tasks or the square footgage of a house in home pricing prediction.  
  
Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. 

Dimensoionality Reduction - Reducing the number of features while preserving important features. A simple example is selecting the area of a house as a feature rather than using width and length seperately. Other examples include singular value decomposition, variational auto-encoders, and t-SNE (for visualizations), and max pooling layers for CNNs.

### Machine learning models and applications

[Neural Nets](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md) - A primer on neural networks.  NNs are a suitable model for fixed input features.  

[Recurrent Neural Nets](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md) - A primer on recurrent neural networks. RNNs are a suitable model for sequences of information.   

[Deep Learning for NLP](https://github.com/andrewt3000/DL4NLP/blob/master/README.md) State of the art deep learning models and nlp applications such as sentiment analysis, translation and dialog generation.  

### Computer Vision
[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) CNNs basics. CNNS  are suitable models for 2d grids of information such as computer vision problems.   

- Image classification: [res net](https://arxiv.org/abs/1512.03385), [Inception v4](https://arxiv.org/abs/1602.07261), [dense net](https://arxiv.org/abs/1608.06993)   
- Object detection: (with bounding boxes) [yolo v4](https://arxiv.org/abs/2004.10934) (realtime object detection)   
- Instance segmentation: [mask r-cnn](https://arxiv.org/abs/1703.06870)  
- Semantic segmentation:  [U-Net](https://arxiv.org/abs/1505.04597)  

### Transfer learning
[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) - storing knowledge gained while solving one problem and applying it to a different but related problem.

### GANS
Generative adversarial networks - GANs - 2 Neural networks compete against each other. The generative network generates candidates while the discriminative network evaluates them. This technique can generate realistic images. See [GAN](https://arxiv.org/abs/1406.2661), [Lap GAN](https://arxiv.org/abs/1506.05751), [DC GAN](https://arxiv.org/abs/1511.06434), [Big GAN](https://arxiv.org/abs/1809.11096), [StyleGAN](https://arxiv.org/abs/1812.04948)   

### Explainablity
Feature visualization - In computer vision, generating images representative of what neural networks are looking for.   
[TensorFlow Lucid](https://github.com/tensorflow/lucid/),  [Activation Atlas](https://distill.pub/2019/activation-atlas/)  

Feature attribution - In computer vision, determining and representing which pixels contribute to a classification. Example: [Saliency maps](https://arxiv.org/pdf/1312.6034.pdf), [Deconvolution](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf), [CAM](https://arxiv.org/pdf/1512.04150.pdf), [Grad-CAM](https://arxiv.org/abs/1610.02391)  
[tf explain](https://github.com/sicara/tf-explain) - tensorflow visualization library.  
[fast ai heatmap](https://docs.fast.ai/vision.learner.html#_cl_int_plot_top_losses) - uses grad-cam  
  
[Lime](https://github.com/marcotcr/lime) 
