### My Notes on Machine Learning

Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

#### Types of machine learning
1. Supervised learning - learns from labeled training data.  
2. Unsupervised learning - learns from unlabled training data. Examples include Principal component analysis and clustering. 
3. Reinforcement learning - Maximize a reward. An agent interacts with an environment and learns to take action by maximizing a cumulative reward.
4. Transfer learning - storing knowledge gained while solving one problem and applying it to a different but related problem.
5. Semi-Supervised learning - Use a mix mostly unlabeled, with a small labeled subset data.  
6. Self-supervised learnng - A form of unsupervised learning where the model is trained on a task using the data itself to generate supervisory signals, rather than relying on externally-provided labels. (Example: Predict the next word (LLM pretraining) or predicting part of a masked image [simCLR](https://arxiv.org/abs/2002.05709), [MAE](https://arxiv.org/abs/2111.06377), [DINO](https://arxiv.org/abs/2104.14294), [iBOT](https://arxiv.org/abs/2111.07832))  



#### Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)
3. Ranking - predicting the relative order or preference of a set of items contextually. (Example: search engine results, or movie recommendations)

#### Features
Features - are the inputs to a machine learning model. They are the measurable property being observed.  An example of a features is pixel brightness in computer vision tasks or the square footgage of a house in home pricing prediction.  
  
**Feature selection** - is the process of choosing the features. Effective features are discriminating and independent. As an example, for predicting house prices you might choose the square feet and number of floors as features whereas width, length and volume are unsuitable features.  

**Feature engineering** - manual, hand-crafted feature extraction. In deep learning, feature engineering is largely replaced by feature learning where the network figures out features automatically.     

**Feature Encoding** - is converting non-numeric data, like text or categories, into numerical formats such as one-hot encoding or embeddings, such as word embeddings for llms. An encoding is any representation in vector form. An embedding is an encoding where closeness = similarity.     

**Feature scaling** - the process of normalizing the range of numeric features. Common feature scaling techniques include min-max scaling, and standardization (aka z-score normalization).  

**Min-max scaling** squeezes values between a range typically 0 to 1. Min-max scaling is best for uniform distributions such as pixel values in image processing. Warning: If you have an outlier (like a single value of 10,000 when everything else is under 10), Min-Max will crush all your normal data into a tiny, indistinguishable band near 0. Min-max scaling is implemented in scikit learn's [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).   

**Standardization** is appropriate for Gaussian distributions, and centers the data on a mean of zero, and a standard deviation of 1. Standardization is implemented by sickit learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).    

**Dimensionality Reduction** - Transforming data from high to low dimension but retains properties. Examples include singular value decomposition, variational auto-encoders, and t-SNE (for visualizations), and max pooling layers for CNNs.

#### Data
Suprervised learning data is typically split into **training**, **validation** and **test** data.  
An **example** (or **sample**) is a single instance from your dataset.  
In supervised learning data, the correct output label is refered to as ground truth.  
**Data drift** (also known as covariate shift) occurs when the statistical properties of the input data change over time compared to the data the model was trained on. For example, for predicting house prices, inflation might change the home values.  

### Machine learning models and applications

[Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md) - Neural networks are a suitable model for fixed input features.  

[Transformers](https://github.com/andrewt3000/MachineLearning/blob/master/transformer.md) - Transformers are a neural network architecture designed to process sequences (text, images, audio, video) using attemtion mechansim. Transformers were originally described in the 2017 paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Transformers replaced [recurrent neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md) for sequential models.    
Transformers are the architecture for: 
- LLMs such as [llama](https://www.llama.com/), [nanochat]([https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanochat)) 
- Multimodal foundation models (Google Gemini, Open AI GPT-5, Anthropic Claude) 
- Vision Transformers (ViT), and Swin Transformers.





### Computer Vision 
These are common computer vision tasks methods for solving them. CNNs have gone through a hybrid period where people use cnn backbones with vision transformers. However the trend is towards transformers. CNNs are still used on realtime mobile devices.      

[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) CNNS  are suitable models for computer vision problems.   

- Image classification: [res net](https://arxiv.org/abs/1512.03385), [Inception v4](https://arxiv.org/abs/1602.07261), [dense net](https://arxiv.org/abs/1608.06993)   
- Object detection: (with bounding boxes) [yolo v4](https://arxiv.org/abs/2004.10934) (still used for realtime object detection)   
- Instance segmentation: [mask r-cnn](https://arxiv.org/abs/1703.06870)  
- Semantic segmentation:  [U-Net](https://arxiv.org/abs/1505.04597)  
