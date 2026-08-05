# Statistics for Machine Learning

Machine learning is applied statistics: a model estimates patterns from a sample of data and is judged on how well those estimates generalize. This page covers the statistical concepts that show up constantly in ML — distributions, sampling, bias and variance, and hypothesis testing for comparing models.

### Descriptive statistics
- **Mean** (μ) - the average. Sensitive to outliers.
- **Median** - the middle value. Robust to outliers, which is why median error metrics are preferred for heavy-tailed data.
- **Variance** (σ²) - the average squared distance from the mean, measuring spread.
- **Standard deviation** (σ) - the square root of variance, in the same units as the data. [Standardization](README.md#features) (z-score normalization) rescales a feature to mean 0 and standard deviation 1.
- **Percentiles / quantiles** - the value below which a given fraction of the data falls. Used in outlier clipping (e.g. winsorizing at the 1st/99th percentile) and in reporting latency (p95, p99).
- **Correlation** - measures linear association between two variables, from −1 to +1. Highly correlated (collinear) features carry redundant information, which is why square footage plus width, length, and volume make poor feature sets together. Correlation is not causation, and correlation of 0 does not imply independence (the relationship may be nonlinear).

### Distributions
A **probability distribution** describes how likely each value of a random variable is. Distributions you will encounter:

- **Gaussian (normal)** - the bell curve, defined by mean and standard deviation. Many natural quantities are approximately Gaussian, standardization assumes it, and weight [initialization](neuralNets.md#initialization) samples from it.
- **Uniform** - all values in a range equally likely. Pixel intensities are closer to uniform, which is why min-max scaling suits them.
- **Bernoulli** - a single yes/no trial with probability p. The distribution behind binary classification labels.
- **Categorical** - one of K outcomes with probabilities summing to 1. The output of a [softmax](neuralNets.md#softmax) layer.
- **Binomial** - the count of successes in n Bernoulli trials. Useful for reasoning about how many wins to expect from n bets or n test predictions.
- **Power law / heavy-tailed** - rare extreme values dominate (word frequencies, wealth, race payouts). Means are unstable for heavy-tailed data; use medians and quantiles.

The **central limit theorem** says the average of many independent samples is approximately Gaussian regardless of the underlying distribution. This is why averages stabilize as datasets grow, and why estimates from small samples are noisy.

### Sampling and estimation
Training data is a **sample** from a larger **population** (the true data distribution). Everything a model learns is an estimate from that sample.

- **Law of large numbers** - estimates converge to true values as sample size grows. Small validation sets give noisy metric estimates; this is why [split fractions shrink but holdout counts stay fixed](data.md) as datasets grow.
- **Sampling bias** - the sample doesn't represent the population (e.g. training a model only on races from one track). No amount of data cures a biased sampling process.
- **i.i.d. assumption** - standard ML assumes examples are independent and identically distributed. Time series and race data violate independence, which is why temporal train/test splits are required to avoid lookahead [data leakage](data.md).
- **Confidence intervals** - a range that quantifies uncertainty in an estimate. A model's "accuracy of 71%" measured on 200 examples is really 71% ± several points; report intervals when validation sets are small.

### Bias and variance
The **bias-variance tradeoff** decomposes generalization error into two sources:

- **Bias** - error from a model too simple to capture the pattern. High bias = [underfitting](neuralNets.md#regularization).
- **Variance** - error from a model too sensitive to the particular training sample; a different sample would produce a very different model. High variance = overfitting.

Increasing model capacity lowers bias and raises variance. [Regularization](neuralNets.md#regularization), more data, and ensembling reduce variance. Bagging (random forests) is explicitly a variance-reduction technique; [boosting](gbm.md) primarily reduces bias.

(Modern deep learning complicates the classic picture: very large networks can fit training data perfectly yet still generalize, the "double descent" phenomenon.)

### Likelihood
The **likelihood** is the probability of the observed data as a function of the model's parameters. **Maximum likelihood estimation (MLE)** picks the parameters that make the observed data most probable.

Most standard loss functions are negative log likelihoods in disguise:
- Minimizing **mean squared error** = MLE under Gaussian noise.
- Minimizing **cross entropy** = MLE for Bernoulli/categorical outcomes.

This is why cross entropy is the principled loss for classification rather than an arbitrary choice, and why log likelihood on held-out data is a standard way to compare probabilistic models.

### Hypothesis testing
Used when comparing models or claiming an improvement is real rather than noise.

- The **null hypothesis** is the default assumption (e.g. "model A and model B perform the same").
- A **p-value** is the probability of seeing a difference at least this large if the null hypothesis were true. Small p-value = the observed difference is unlikely to be luck. Conventionally p < 0.05 is "significant," but the threshold is arbitrary.
- A p-value is **not** the probability the null hypothesis is true, and statistical significance is not practical significance — with enough data, trivial differences become "significant."
- **Multiple comparisons** - test enough model variants and one will look significant by chance. Evaluating many experiments against the same validation set gradually overfits to it; this is a form of [data leakage](data.md) at the research-process level.
- For comparing two models in practice: run multiple seeds and compare the distributions of scores, or use a paired test (paired t-test, or bootstrap resampling of the test set) rather than comparing two single numbers.

### Bayes' theorem
**Bayes' theorem** updates a prior belief with evidence:

P(A|B) = P(B|A) · P(A) / P(B)

- Explains **base rate effects**: a 99%-accurate test for a rare (1 in 10,000) condition still produces mostly false positives, because the prior is so low. The same arithmetic is why [accuracy is misleading on imbalanced data](neuralNets.md#metrics).
- **Naive Bayes** classifiers apply the theorem directly with an independence assumption.
- The Bayesian view of ML treats parameters as distributions rather than point estimates; regularization corresponds to a prior on the weights (L2 = Gaussian prior, L1 = Laplace prior).

### References / tutorials
- [Seeing Theory](https://seeing-theory.brown.edu/) - visual, interactive introduction to probability and statistics
- [StatQuest](https://www.youtube.com/@statquest) - short videos on statistics and ML fundamentals
- 2016 Wasserstein & Lazar, the ASA's statement on p-values [The ASA Statement on p-Values: Context, Process, and Purpose](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108)
- [3Blue1Brown: Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) - visual intuition for Bayes' theorem
