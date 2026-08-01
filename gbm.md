### Gradient Boosted Decision Trees (GBDT)
**Gradient boosted decision trees** are an ensemble of shallow decision trees built sequentially, where each new tree is trained to correct the errors of the trees before it. Despite the dominance of neural networks elsewhere, GBDTs remain state of the art for most **tabular data** problems (structured rows and columns of numeric and categorical features) and typically win with less data, less tuning, and less compute [1](#gbdt-references).

- A **decision tree** predicts by splitting the data with a series of if/then rules on feature values. Individual trees are interpretable but overfit easily.
- **Ensembles** combine many weak trees into one strong model. The two main strategies are bagging and boosting.
- **Bagging** (e.g. **random forests**) trains many deep trees independently on random subsets of data and features, then averages them. Averaging reduces variance.
- **Boosting** trains shallow trees sequentially. Each tree is fit to the **residual errors** (more precisely, the negative gradient of the loss) of the current ensemble, so the model improves iteratively. Predictions are the sum of all trees, each scaled by the learning rate.

Because each tree fits the gradient of a loss function, GBDTs work with any differentiable loss: squared error for regression, log loss for classification, and ranking losses (e.g. LambdaRank) for learning to rank.

#### Key hyperparameters
- **Number of trees** (boosting rounds) - more trees fit the data better but eventually overfit. Typically controlled with early stopping on a validation set.
- **Learning rate** (shrinkage) - scales each tree's contribution. Lower values (0.01–0.1) need more trees but generalize better.
- **Tree depth / number of leaves** - controls the complexity of each tree. Boosted trees are kept shallow (depth 3–8), unlike the deep trees in random forests.
- **Subsampling** of rows and features per tree adds randomness that regularizes the model.

#### Libraries
- [XGBoost](https://xgboost.readthedocs.io/) - popularized the modern regularized GBDT.
- [LightGBM](https://lightgbm.readthedocs.io/) - grows trees leaf-wise instead of level-wise and uses histogram-based splits, making it faster on large datasets.
- [CatBoost](https://catboost.ai/) - handles categorical features natively with ordered target encoding.
- scikit-learn [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

#### Why trees beat neural networks on tabular data
Tabular features are heterogeneous (different units, scales, and meanings) and often have irregular, non-smooth relationships to the target. Trees handle this naturally: splits are invariant to feature scaling, robust to outliers and uninformative features, and can model sharp thresholds that smooth neural network functions approximate poorly [1](#gbdt-references). Neural networks tend to win on tabular data only when datasets are very large, or when inputs include unstructured data (text, images, variable-length sequences) that require learned representations.

#### GBDT References
1. 2022 Grinsztajn et al. [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
2. 2001 Friedman, the original gradient boosting paper [Greedy Function Approximation: A Gradient Boosting Machine]([https://jerryfriedman.su.domains/ftp/trebst.pdf](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)
3. 2016 XGBoost paper [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
4. 2017 LightGBM paper [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
