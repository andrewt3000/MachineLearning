# My Notes on Machine Learning
These are my notes on machine learning, [neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md), and [transformers](https://github.com/andrewt3000/MachineLearning/blob/master/transformer.md).  


## Machine Learning
Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

### Types of machine learning
Machine learning problems are categorize as discriminative or generative.  
1. Supervised learning - learns from labeled training data.
   - svm, knn, random forests, Gradient Boosting Machines, [neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md)
3. Unsupervised learning - learns from unlabled training data.
   - principal component analysis, clustering. 
4. Reinforcement learning - Maximize a reward. An agent interacts with an environment and learns to take action by maximizing a cumulative reward.
   - Q-Learning, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO)
6. Transfer learning - storing knowledge gained while solving one problem and applying it to a different but related problem.
7. Semi-Supervised learning - Use a mix mostly unlabeled, with a small labeled subset data.  
8. Self-supervised learning - A form of unsupervised learning where the model is trained on a task using the data itself, rather than labels. 
   - Autoregressive LLM pretraining (next word prediction), masked image modeling such as [MAE masked autoencoders](https://arxiv.org/abs/2111.06377).  

### Machine learning problems
1. Regression - predicting a continuous value attribute (Example: house prices)
2. Classification - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)
   - Classification is further categorized as binary or multi-class classificaition.  
4. Ranking - predicting the relative order or preference of a set of items contextually. (Example: search engine results, or movie recommendations)

### Data
Data is typically split into **training**, **validation** and **test** data. Typical mix is:
- 70 - 80% training data
- 10 - 15% validation data
- 10 - 15% test data
  
**Training data** is the data used to learn.  
**Validation data** is evaluated while the model is training and indicates if the model is generalizing.  
**Test data** is evaluated after you training to indicate the model's accuracy.  

An **example** (or **sample**) is a single instance from your dataset.  

**Labels** are the correct outputs in categorical supervised learning data, also refered to as **ground truth**.  

**Features** are the inputs to a machine learning model. See [below](#features)    

**Data augmentation** - artificially increase the diversity and size of a training dataset without actually collecting new data. An example of data augmentation for images is mirroring, flipping, rotating, and translating your images to create new examples.  


**Data leakage** is when information from outside the training dataset (such as the target variable or the validation/test set) accidentally contaminates the training process.  

**Data drift** (also known as covariate shift) occurs when the statistical properties of the input data change over time compared to the data the model was trained on.  

**Concept drift** occurs when there is a change in input-output relationship over time compared to the data the model was trained on.  

### Features
**Features** are the inputs to a machine learning model. Features are the measurable property being observed.  An example of a features is pixel brightness in computer vision tasks or the square footgage of a house in home pricing prediction.  
  
**Feature selection** is the process of choosing the features. Effective features are discriminating and independent. As an example, for predicting house prices you might choose the square feet and number of floors as features whereas width, length and volume are unsuitable features.  

**Feature engineering** is manual, hand-crafted feature extraction. In deep learning, feature engineering is largely replaced by feature learning where the network figures out features automatically.     

**Feature Encoding** is converting non-numeric data, like text or categories, into numerical formats such as one-hot (or muli-hot) encodings or embeddings, such as word embeddings for llms.  

An **encoding** is any representation in vector form.  

An **embedding** is an encoding where numerical closeness indicates similarity.     

**Feature scaling** is the process of normalizing the range of numeric features. Common feature scaling techniques include min-max scaling, and standardization (aka z-score normalization).  

**Min-max scaling** squeezes values between a range typically 0 to 1. Min-max scaling is best for uniform distributions such as pixel values in image processing. Warning: If you have an outlier (like a single value of 10,000 when everything else is under 10), Min-Max will crush all your normal data into a tiny, indistinguishable band near 0. Min-max scaling is implemented in scikit learn's [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).   

**Standardization** is appropriate for Gaussian distributions, and centers the data on a mean of zero, and a standard deviation of 1. Standardization is implemented by sickit learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).    

**Dimensionality Reduction** is transforming data from high to low dimension but retaining the properties. Examples include singular value decomposition, variational auto-encoders, and t-SNE (for visualizations), and max pooling layers for CNNs.



### Machine learning models and applications

[Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md) - Neural networks are a suitable model for fixed input features.  

[Transformers](https://github.com/andrewt3000/MachineLearning/blob/master/transformer.md) - Transformers are a neural network architecture designed to process sequences using attemtion mechansim. Transformers are the architecture for large language models and vision transformers.  Transformers largely replaced [recurrent neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md) for sequential models.    


### Computer Vision 
These are common computer vision tasks methods for solving them. CNNs have gone through a hybrid period where it's common to use cnn backbones with vision transformers. However the trend is towards transformers. CNNs are still used on realtime and mobile devices because they require less resources.      

[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) CNNS are suitable models for computer vision problems.   

- Image classification: [res net](https://arxiv.org/abs/1512.03385), [Inception v4](https://arxiv.org/abs/1602.07261), [dense net](https://arxiv.org/abs/1608.06993)   
- Object detection: (with bounding boxes) [ultralytics yolo](https://github.com/ultralytics/ultralytics) (In 2026, sota for realtime/edge object detection)   
- Instance segmentation: [mask r-cnn](https://arxiv.org/abs/1703.06870)  
- Semantic segmentation:  [U-Net](https://arxiv.org/abs/1505.04597)
