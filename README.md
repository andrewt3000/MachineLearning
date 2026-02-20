### My Notes on Machine Learning

Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

#### Types of machine learning
1. Supervised learning - learns from labeled training data.  
2. Unsupervised learning - learns from unlabled training data. Examples include Principal component analysis and clustering. 
3. Reinforcement learning - Maximize a reward. An agent interacts with an environment and learns to take action by maximizing a cumulative reward.
4. [Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) - storing knowledge gained while solving one problem and applying it to a different but related problem.
5. [Semi-Supervised learning](https://en.wikipedia.org/wiki/Weak_supervision) - Use a mix mostly unlabeled, with a small labeled subset data.  
6. [Self-supervised learnng](https://en.wikipedia.org/wiki/Self-supervised_learning) - A form of unsupervised learning where the model is trained on a task using the data itself to generate supervisory signals, rather than relying on externally-provided labels. (Example: Predict the next word (LLM pretraining) or predicting part of a masked image [MAE](https://arxiv.org/abs/2111.06377), [DINO](https://arxiv.org/abs/2104.14294), [iBOT](https://arxiv.org/abs/2111.07832))  



#### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)

#### Input features
[Features](https://en.wikipedia.org/wiki/Feature_(machine_learning)) - are the inputs to a machine learning model. They are the measurable property being observed.  An example of a features is pixel brightness in computer vision tasks or the square footgage of a house in home pricing prediction.  
  
Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. 

Dimensionality Reduction - Reducing the number of features while preserving important features. A simple example is selecting the area of a house as a feature rather than using width and length seperately. Other examples include singular value decomposition, variational auto-encoders, and t-SNE (for visualizations), and max pooling layers for CNNs.

#### Data
In suprervised learning data is typically split into training, validation and test data.  
An example is a single instance from your dataset.  

### Machine learning models and applications

[Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md) - Neural networks are a suitable model for fixed input features.  

[Transformers](https://github.com/andrewt3000/MachineLearning/blob/master/transformer.md) - Transformers are a neural network architecture designed to process sequences (text, images, audio, video) using a mechanism called attention. Replacing [recurrent neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md).  Originally described in 2017 paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
Transformers are the architecture for 
- LLMs such as [chat gpt](https://chat.openai.com/), [nanogpt](https://github.com/karpathy/nanoGPT) 
- Vision Transformers (ViT)
- Multimodal foundation models (CLIP, SigLIP, OpenAI’s Vision models, Gemini, etc.)





### Computer Vision 
These are common computer vision tasks (obsolete) methods for solving them.  

[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) CNNS  are suitable models for computer vision problems.   

- Image classification: [res net](https://arxiv.org/abs/1512.03385), [Inception v4](https://arxiv.org/abs/1602.07261), [dense net](https://arxiv.org/abs/1608.06993)   
- Object detection: (with bounding boxes) [yolo v4](https://arxiv.org/abs/2004.10934) (still used for realtime object detection)   
- Instance segmentation: [mask r-cnn](https://arxiv.org/abs/1703.06870)  
- Semantic segmentation:  [U-Net](https://arxiv.org/abs/1505.04597)  
