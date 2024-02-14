### My Notes on Machine Learning

Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

#### 3 Types of machine learning
1. Supervised learning - Minimize an error function using labeled training data.  
2. Unsupervised learning - Find patterns using unlabled training data. Examples include Principal component analysis and clustering. 
3. Reinforcement learning - Maximize a reward. An agent interacts with an environment and learns to take action by maximizing a cumulative reward.   

#### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)

#### Input features
[Features](https://en.wikipedia.org/wiki/Feature_(machine_learning)) - are the inputs to a machine learning model. They are the measurable property being observed.  An example of a features is pixel brightness in computer vision tasks or the square footgage of a house in home pricing prediction.  
  
Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. 

Dimensoionality Reduction - Reducing the number of features while preserving important features. A simple example is selecting the area of a house as a feature rather than using width and length seperately. Other examples include singular value decomposition, variational auto-encoders, and t-SNE (for visualizations), and max pooling layers for CNNs.

#### Data
In suprervised learning data is typically split into training, validation and test data.  

An example is a single instance from your dataset.  

### Machine learning models and applications

[Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md) - Neural networks are a suitable model for fixed input features.  

[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) CNNS  are suitable models for computer vision problems.   

Transformers - are designed to handle sequential data, like text, in a manner that allows for much more parallelization than previous models like [recurrent neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md).  See 2017 paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
Transformer is the architecture for [chat gpt](https://chat.openai.com/). Example [nanogpt](https://github.com/karpathy/nanoGPT)      

### Computer Vision
These are common computer vision tasks and state of the art methods for solving them.  

- Image classification: [res net](https://arxiv.org/abs/1512.03385), [Inception v4](https://arxiv.org/abs/1602.07261), [dense net](https://arxiv.org/abs/1608.06993)   
- Object detection: (with bounding boxes) [yolo v4](https://arxiv.org/abs/2004.10934) (realtime object detection)   
- Instance segmentation: [mask r-cnn](https://arxiv.org/abs/1703.06870)  
- Semantic segmentation:  [U-Net](https://arxiv.org/abs/1505.04597)  

### NLP Natural Language Processing
[Deep Learning for NLP](https://github.com/andrewt3000/DL4NLP/blob/master/README.md) Deep learning models and nlp applications such as sentiment analysis, translation and dialog generation.  

### Transfer learning
[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) - storing knowledge gained while solving one problem and applying it to a different but related problem.

### Explainablity
Feature visualization - In computer vision, generating images representative of what neural networks are looking for.   
[TensorFlow Lucid](https://github.com/tensorflow/lucid/),  [Activation Atlas](https://distill.pub/2019/activation-atlas/)  

Feature attribution - In computer vision, determining and representing which pixels contribute to a classification. Example: [Saliency maps](https://arxiv.org/pdf/1312.6034.pdf), [Deconvolution](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf), [CAM](https://arxiv.org/pdf/1512.04150.pdf), [Grad-CAM](https://arxiv.org/abs/1610.02391)  
[tf explain](https://github.com/sicara/tf-explain) - tensorflow visualization library.  
[fast ai heatmap](https://docs.fast.ai/vision.learner.html#_cl_int_plot_top_losses) - uses grad-cam  
  
[Lime](https://github.com/marcotcr/lime) 
