# Statistics for Machine Learning

Machine learning is applied statistics: a model estimates patterns from a sample of data and is judged on how well those estimates generalize. This page covers the statistical concepts that show up constantly in ML — distributions, sampling, bias and variance, and hypothesis testing for comparing models.

### Descriptive statistics
- **Mean** (μ) - the average. Sensitive to outliers.
  
  $$\mu = \frac{1}{N}\sum_{i=1}^{N}x_i$$
  
- **Median** - the middle value. Robust to outliers, which is why median error metrics are preferred for heavy-tailed data.
- **Variance** (σ²) - the average squared distance from the mean, measuring spread.

  $$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2 \qquad $$

- **Standard deviation** (σ) - the square root of variance, in the same units as the data. [Standardization](data.md) (z-score normalization) rescales a feature to mean 0 and standard deviation 1.
   
  $$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$$

- **Percentiles / quantiles** - the value below which a given fraction of the data falls. Used in outlier clipping (e.g. winsorizing at the 1st/99th percentile) and in reporting latency (p95, p99).
- **Correlation** - measures linear association between two variables, from −1 to +1. Highly correlated (collinear) features carry redundant information, which is why square footage plus width, length, and volume make poor feature sets together. Correlation is not causation, and correlation of 0 does not imply [independence](#independence) (the relationship may be nonlinear).

### Distributions
A **probability distribution** describes how likely each value of a random variable is. 

Distributions are either **discrete** or **continuous**:
- **Discrete** distributions describe outcomes you can count — a coin flip, a class label, a number of wins. Each outcome has a probability, given by a **probability mass function (PMF)**, and the probabilities sum to 1.
- **Continuous** distributions describe outcomes on a continuous scale — a height, a time, a price. Any single exact value has probability 0, so instead a **probability density function (PDF)** gives density, and probability is the *area* under the curve over a range: P(a ≤ X ≤ b) is the [integral](calculus.md#integrals) of the PDF from a to b. The total area is 1.

The practical consequence is how you sum: discrete distributions sum over outcomes, continuous ones integrate. This is the difference between a bar chart and a smooth curve.

Distributions you will encounter:  

- **Gaussian (normal)** - (*continuous*) the bell curve, defined by mean and standard deviation. Many natural quantities are approximately Gaussian, standardization assumes it, and weight [initialization](neuralNets.md#initialization) samples from it.
- **Uniform** - (*continuous*) all values in a range equally likely. Pixel intensities are closer to uniform, which is why min-max scaling suits them.
- **Bernoulli** - (*discrete*) a single yes/no trial with probability p. The distribution behind binary classification labels.
- **Categorical** - (*discrete*) one of K outcomes with probabilities summing to 1. The output of a [softmax](neuralNets.md#softmax) layer.
- **Binomial** - (*discrete*) the count of successes in n Bernoulli trials. Useful for reasoning about how many wins to expect from n bets or n test predictions.
- **Power law / heavy-tailed** - (both. Zipf is *discrete*, Pareto is *continuous*) rare extreme values dominate (word frequencies, wealth, race payouts). Means are unstable for heavy-tailed data; use medians and quantiles.

The **central limit theorem** says the average of many independent samples is approximately Gaussian regardless of the underlying distribution. This is why averages stabilize as datasets grow, and why estimates from small samples are noisy.

<img width="680" height="735" alt="distributions" src="https://github.com/user-attachments/assets/df2efc22-64c1-4892-bead-04cdb22f92e5" />


### Independence
Two events are **independent** if knowing one tells you nothing about the other:

$$P(A \cap B) = P(A)P(B) \qquad \text{equivalently} \qquad P(A|B) = P(A)$$

The conditional form is the more intuitive one: conditioning on B doesn't change the probability of A. Successive coin flips are independent; drawing cards without replacement is not, because each draw changes what remains. Misjudging independence has a name in both directions: the **gambler's fallacy** treats independent events as if they correct themselves ("red is due"), while the **hot hand fallacy** treats them as if they streak. Assuming independence where it doesn't hold is the error that matters more in ML — see [when the assumption breaks](#when-the-assumption-breaks).  

**Conditional independence** is the weaker and more useful version: A and B may be dependent overall but independent once you know C. Ice cream sales and drownings are correlated, but conditional on temperature they are roughly independent — the "common cause" pattern.

#### Why it matters in ML
- **Likelihoods factor.** If examples are independent, the probability of the whole dataset is the product of the individual probabilities — and taking logs turns that product into a *sum*. This is why the loss over a dataset is the sum (or average) of per-example losses, and why mini-batches give unbiased gradient estimates. Nearly all of ML's optimization machinery rests on this assumption.
- **Errors accumulate slowly.** Averaging n independent estimates reduces standard error by $\sqrt{n}$ ([central limit theorem](#distributions)). If the samples are correlated, the effective sample size is smaller than n and the true uncertainty is larger than the formula suggests.
- **Ensembles need diversity.** Averaging models only reduces variance to the extent their errors are independent. Bagging deliberately decorrelates trees via random subsampling; averaging ten identical models buys nothing.
- **Naive Bayes** assumes features are conditionally independent given the label. This is usually false, hence "naive" — yet the classifier often works anyway, because the decision boundary can be right even when the probabilities are miscalibrated.

#### Independence vs correlation
Independence implies zero correlation, but not the reverse. Correlation only measures *linear* association: $y = x^2$ over a symmetric range has correlation ≈ 0 while y is completely determined by x. Zero correlation is a weak check; independence is a strong claim.

#### When the assumption breaks
Real data is often not independent, and the failure is usually invisible until results don't replicate:
- **Time series** - today's value depends on yesterday's (autocorrelation). Random train/test splits leak future into past; use temporal splits.
- **Grouped data** - multiple rows from the same entity (patients in a hospital, horses in a race, users in a session) are correlated with each other. Random splits put related rows on both sides of the split, inflating validation scores. Split by group instead.
- **Repeated measurements** - 1,000 frames from one video are not 1,000 independent images. The effective sample size is closer to the number of videos.

The consequence is nearly always the same: uncertainty is underestimated, and models look better in validation than they perform in deployment.
### Sampling and estimation
Training data is a **sample** from a larger **population** (the true data distribution). Everything a model learns is an estimate from that sample.

- **Law of large numbers** - estimates converge to true values as sample size grows. Small validation sets give noisy metric estimates; this is why [split fractions shrink but holdout counts stay fixed](data.md) as datasets grow.
- **Sampling bias** - the sample doesn't represent the population (e.g. training a model only on races from one track). No amount of data cures a biased sampling process.
- **i.i.d. assumption** - standard ML assumes examples are [independent](#independence) and identically distributed. Time series data may violate [independence](#independence), which is why temporal train/test splits are required to avoid lookahead [data leakage](data.md).
- **Standard error (SE)** - the standard deviation of an *estimate* (as opposed to the data). It shrinks with sample size, which is why bigger validation sets give more trustworthy metrics.
  - For a mean: $SE = \frac{\sigma}{\sqrt{n}}$ where σ is the sample standard deviation.
  - For a proportion such as accuracy: $SE = \sqrt{\frac{p(1-p)}{n}}$
- **Confidence intervals** - a range that quantifies uncertainty in an estimate, computed as estimate ± z · SE. For a 95% interval, z ≈ 1.96 (the "±2 standard errors" rule of thumb).
  - A model's "accuracy of 71%" measured on 200 examples has $SE = \sqrt{\frac{0.71 \times 0.29}{200}} \approx 0.032$, so the 95% interval is roughly 71% ± 6.3%, or 65% to 77%. A rival model scoring 74% is not meaningfully better.
  - Note the $\sqrt{n}$: halving the interval requires 4× the data.
  - **Interpretation** - strictly, "95% confidence" describes the *procedure*, not this particular interval: if you repeated the experiment many times, 95% of the intervals constructed this way would contain the true value. It does not mean there is a 95% probability the true value lies in the interval you computed — in frequentist statistics the true value is fixed, and the interval is what's random. The intuitive "95% probability it's in here" reading is a **credible interval**, the Bayesian analogue, which requires a prior. In practice the two often nearly coincide, and the distinction rarely changes a decision — but it is the same category error as reading a p-value as "the probability the null hypothesis is true."

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
