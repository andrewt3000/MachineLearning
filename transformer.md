A Transformer is a neural network architecture built entirely around attention. It was introduced in “Attention Is All You Need” (Vaswani et al., 2017). Transformers power models like BERT, GPT-3/4/5, etc.

## 🔢 Input Representation
- The model receives a sequence of tokens (words/subwords), each mapped to a vector embedding.
- A positional encoding is added to each token so the model knows the order.
- Together, these embeddings form a matrix where each row is a token vector.

## 🔍 Self-Attention
Self-attention computes how much each token should “pay attention” to every other token in the sequence.
To do this, the model learns 3 projection matrices:

These transform each token vector into:
- Q (Query) – represents the current token.
- K (Key) – represents the tokens being compared against.
- V (Value) – carries the actual information to mix together.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 🧠 Multi-Head Attention
Instead of one attention operation, the model uses h attention heads:
- Each head learns different relationships (syntax, long-range dependencies, semantics, etc.).
- Heads are concatenated and combined into a single output.

This lets the model analyze many patterns in parallel.

## 🏛️ Encoder and Decoder
The original Transformer has two parts:
### Encoder
- Uses self-attention
- Produces contextualized representations
- Used for understanding
- BERT = encoder-only

### Decoder
- Uses masked self-attention (can’t see future tokens)
- Attends to encoder output
- Used for text generation
- GPT / ChatGPT = decoder-only

## 🎯 Training Tricks
Label Smoothing
- Makes the model less overconfident
- Helps generalization and stability

Residual Connections + Layer Norm
- Help gradients flow
- Stabilize deep models

Feed-Forward Networks
- After attention, each token passes through a small MLP
- Adds non-linearity and additional modeling capacity

## ✨ Why Transformers Work So Well
- Parallelizable (no recurrence) → trains fast
- Long-range dependencies captured easily
- Multi-head attention discovers rich relationships
- Scales to billions/trillions of parameters

## Final Concept Summary
- Inputs = token embeddings + positions
- Self-attention projects tokens into Q, K, V
- Multi-head attention runs several attentions in parallel
- Encoder = understanding; decoder = generation
- Main formula: attention = softmax(similarity) × values
- Label smoothing improves training stability
