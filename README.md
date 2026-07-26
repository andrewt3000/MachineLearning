# Machine Learning Overview
- [data and features](data.md)
- [reinforcement learning](rl.md)
- [neural networks](neuralNets.md)
- [transformers and llms](transformer.md)
- [vision transformers](cv.md)
- Legacy: [cnn](cnn4Images.md), [rnn](rnn.md)

Machine Learning is a sub-field of artificial intelligence that uses data to train predictive models.  

## Types of machine learning

1. **Supervised learning** - learns from **labeled** training data.
   - svm, knn, random forests, gradient boosting machines, [neural networks](neuralNets.md)
2. **Unsupervised learning** - learns from unlabeled training data.
   - principal component analysis, clustering. 
3. [**Reinforcement learning**](rl.md) - An **agent** interacts with an **environment** and learns to take **action** by maximizing a cumulative **reward**.
   - Q-Learning, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO)
4. **Semi-Supervised learning** - trains on a mix of mostly unlabeled with a small labeled subset data.  
5. **Self-supervised learning** - A form of unsupervised learning where training labels are constructed automatically from the data itself. 
   - Autoregressive LLM pretraining (next word prediction), and masked image modeling.
  
## Machine learning problems
1. **Regression** - predicting a continuous value attribute.
   - Example: predicting house prices
2. **Classification** - predicting a discrete value. 
   - Classification is further categorized as binary or multi-class classification.
   - Binary Example: predicting pass or fail, benign or malignant, spam or not spam, hot dog or not hot dog :-)
   - Multi-Class Example: Handwritten Digit Recognition (0 through 9) [mnist](https://huggingface.co/datasets/ylecun/mnist), 1,000 classes [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)  
3. **Ranking** - predicting the relative order or preference of a set of items contextually.
   - Example: search engine results, or movie recommendations 

Models that predict labels from inputs (as in the problems above) are called **discriminative**; models that learn the data distribution to synthesize new samples (diffusion, LLMs) are **generative**.

### Transfer learning
Transfer learning is storing knowledge gained while solving one problem and applying it to a different but related problem.
   - **fine tuning** is additional training to a base model for a specific task.  
   - **LoRA** (Low-Rank Adaptation) is a fine-tuning method that freezes the base model's weights and trains small low-rank matrices that are added to existing layers.

### Metric Learning
**Metric learning** trains a model to produce embeddings where distance reflects similarity. A **siamese network** passes two inputs through identical networks with shared weights and compares the resulting embeddings. Trained with **contrastive loss** (pull matching pairs together, push non-matching pairs apart) or **triplet loss** (anchor, positive, negative). 
- Example: face verification, signature verification
