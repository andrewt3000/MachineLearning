# Overview of Machine Learning
- [data and features](https://github.com/andrewt3000/MachineLearning/blob/master/data.md)
- [neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md)
- [transformers and llms](https://github.com/andrewt3000/MachineLearning/blob/master/transformer.md)
- [vision transformers](https://github.com/andrewt3000/MachineLearning/blob/master/cv.md) 

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
   - **fine tuning** is additional training to a base model weight's for a specific task.  
   - **LoRa** (Low-Rank Adaptation) is a fine-tuning method that freezes the base model's weights and trains small low-rank matrices that are added to existing layers.
5. **Semi-Supervised learning** - trains on a mix of mostly unlabeled with a small labeled subset data.  
6. **Self-supervised learning** - A form of unsupervised learning where the model is trained on a task using the data itself, rather than labels. 
   - Autoregressive LLM pretraining (next word prediction), and masked image modeling.  

### Machine learning problems
1. **Regression** - predicting a continuous value attribute (Example: house prices)
2. **Classification** - predicting a discrete value. (Example: pass or fail, hot dog/not hot dog)
   - Classification is further categorized as binary or multi-class classificaition.  
3. **Ranking** - predicting the relative order or preference of a set of items contextually.
   - Example: search engine results, or movie recommendations 

## Machine Learning Problems
Machine learning problems are categorized as discriminative or generative.  
- **discriminative** models predict labels from inputs (classification, regression).
- **generative** models learn to synthesize new samples (diffusion models, LLMs).  

### Metric Learning
**Metric learning** trains a model to produce embeddings where distance reflects similarity. A **siamese network** passes two inputs through identical networks with shared weights and compares the resulting embeddings. Trained with **contrastive loss** (pull matching pairs together, push non-matching pairs apart) or triplet loss (anchor, positive, negative). Used for face verification, signature verification, one-shot learning, and sentence embeddings. 

### Edge Computing
[Convolutional Neural Networks](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md) CNNs were historically popular but are largely being replaced by transformer models and VLMs. CNNs have gone through a hybrid period where it's common to use cnn backbones with vision transformers. CNNs are still used on realtime and mobile devices because they require less resources. See [yolo](https://github.com/ultralytics/ultralytics) library for real time edge device CV.    
