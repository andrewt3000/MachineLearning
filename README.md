# My Notes on Machine Learning
These are my notes on machine learning, [neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md), and [transformers](https://github.com/andrewt3000/MachineLearning/blob/master/transformer.md).  


## Machine Learning
Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

### Types of machine learning

1. **Supervised learning** - learns from **labeled** training data.
   - svm, knn, random forests, gradient boosting machines, [neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md)
2. **Unsupervised learning** - learns from unlabled training data.
   - principal component analysis, clustering. 
3. [**Reinforcement learning**](rl.md) - An **agent** interacts with an **environment** and learns to take **action** by maximizing a cumulative **reward**.
   - Q-Learning, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO)
4. **Transfer learning** - storing knowledge gained while solving one problem and applying it to a different but related problem.
   - **fine tuning** is additional training to a base model for a specific task.  
   - **lora** (Low-Rank Adaptation) is an add on for base models or stable diffusion model that adds ability for a specific task.  
5. **Semi-Supervised learning** - Use a mix mostly unlabeled, with a small labeled subset data.  
6. **Self-supervised learning** - A form of unsupervised learning where the model is trained on a task using the data itself, rather than labels. 
   - Autoregressive LLM pretraining (next word prediction), and masked image modeling.  

### Machine learning problems
1. **Regression** - predicting a continuous value attribute (Example: house prices)
2. **Classification** - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)
   - Classification is further categorized as binary or multi-class classificaition.  
3. **Ranking** - predicting the relative order or preference of a set of items contextually.
   - Example: search engine results, or movie recommendations

### Data
Data is typically split into **training**, **validation** and **test** data. Typical mix is:
| Dataset size | Split (train / val / test) | Val size | Test size | Notes |
|---|---|---|---|---|
| ≤ 10k examples | 70 / 15 / 15 (or 60 / 20 / 20) | ~1.5k | ~1.5k | Consider k-fold cross-validation instead |
| ~100k examples | 90 / 5 / 5 | ~5k | ~5k | Val/test now large enough for stable estimates |
| ~1M examples | 98 / 1 / 1 | ~10k | ~10k | The canonical big-data split |
| ~10M examples | 99.8 / 0.1 / 0.1 | ~10k | ~10k | Hold-out counts stay fixed; fraction shrinks |
| 100M+ examples | ~99.99 / fixed / fixed | ~10k–50k | ~10k–50k | Fixed-size held-out samples, well under 1% |

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



### Machine learning models

[Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md) - Neural networks are machine learning models suitable for fixed inputs and they fully connected layers in more complex models.  

[Transformers](https://github.com/andrewt3000/MachineLearning/blob/master/transformer.md) - Transformers are a neural network architecture designed to process sequences using attention mechansim. Transformers are the architecture for large language models and vision transformers.  Transformers largely replaced [recurrent neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md) for sequential models.    

## Machine Learing Problems

Machine learning problems are categorize as **discriminative** or **generative**.  

### Computer Vision 
These are common computer vision tasks methods for solving them. CNNs have gone through a hybrid period where it's common to use cnn backbones with vision transformers. However the trend is towards transformers. CNNs are still used on realtime and mobile devices because they require less resources.     

### Vision Transformers (ViT) Papers
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [MAE Masked autoencoders ](https://arxiv.org/abs/2111.06377)
- [DINO](https://arxiv.org/abs/2104.14294)
- [iBOT](https://arxiv.org/abs/2111.07832)

### Generative Computer Vision
- text 2 image, image 2 image - [latent diffusion](https://arxiv.org/abs/2112.10752)

### Realtime discriminative computer vision
[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) CNNS are suitable models for computer vision problems.   

- Image classification: [res net](https://arxiv.org/abs/1512.03385), [Inception v4](https://arxiv.org/abs/1602.07261), [dense net](https://arxiv.org/abs/1608.06993)   
- Object detection: (with bounding boxes) [ultralytics yolo](https://github.com/ultralytics/ultralytics) (In 2026, sota for realtime/edge object detection)   
- Instance segmentation: [mask r-cnn](https://arxiv.org/abs/1703.06870)  
- Semantic segmentation:  [U-Net](https://arxiv.org/abs/1505.04597)
