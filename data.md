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
**Test data** is evaluated after training to indicate the model's accuracy.  

An **example** (or **sample**) is a single instance from your dataset.  

**Labels** are the correct outputs in supervised learning data, also refered to as **ground truth**.  

**Features** are the inputs to a machine learning model. See [below](#features)    

**Data augmentation** - artificially increase the diversity and size of a training dataset without actually collecting new data. An example of data augmentation for images is mirroring, flipping, rotating, and translating your images to create new examples.  


**Data leakage** is when information from outside the training dataset (such as the target variable or the validation/test set) accidentally contaminates the training process.  

**Data drift** (also known as covariate shift) occurs when the statistical properties of the input data change over time compared to the data the model was trained on.  

**Concept drift** occurs when there is a change in input-output relationship over time compared to the data the model was trained on.  

### Features
**Features** are the inputs to a machine learning model. Features are the measurable property being observed.  An example of a feature is pixel brightness in computer vision tasks or the square footage of a house in home pricing prediction.  
  
**Feature selection** is the process of choosing the features. Effective features are discriminating and independent. As an example, for predicting house prices you might choose the square feet and number of floors as features whereas width, length and volume are unsuitable features.  

**Feature engineering** is manual, hand-crafted feature extraction. In deep learning, feature engineering is largely replaced by feature learning where the network figures out features automatically.     

**Feature Encoding** is converting non-numeric data, like text or categories, into numerical formats such as one-hot (or multi-hot) encodings or embeddings, such as word embeddings for llms.  

An **encoding** is any representation in vector form.  

An **embedding** is an encoding where numerical closeness indicates similarity.     

**Feature scaling** is the process of normalizing the range of numeric features. Common feature scaling techniques include min-max scaling, and standardization (aka z-score normalization).  

**Min-max scaling** squeezes values between a range typically 0 to 1. Min-max scaling is best for uniform distributions such as pixel values in image processing. Warning: If you have an outlier (like a single value of 10,000 when everything else is under 10), Min-Max will crush all your normal data into a tiny, indistinguishable band near 0. Min-max scaling is implemented in scikit learn's [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).   

**Standardization** is appropriate for Gaussian distributions, and centers the data on a mean of zero, and a standard deviation of 1. Standardization is implemented by sickit learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).    

**Dimensionality Reduction** is transforming data from high to low dimension but retaining the properties. Examples include PCA, singular value decomposition, variational auto-encoders, and t-SNE (for visualizations). Downsampling such as max pooling layers for CNNs is simply removing dimensions regardless of whether they retain properties or not.


