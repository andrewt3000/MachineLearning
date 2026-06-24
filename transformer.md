A Transformer is a neural network architecture built with attention layers. It was introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). Transformers are the large language models in multimodal foundation models such as Google Gemini, Open AI Chat GPT, Anthropic Claude.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/transformer.png" height='460px' width='326px'/>  

### Original Transformer
The original “attention is all you need” paper was for language translation. The input is “encoded” from the original language and then the output is decoded in a second target language. 
The attention paper model has an encoder on the left and a decoder on the right.
Both encoder and decoder contain “Multi-head attention” MHA layers. [pytorch implementation](https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html) 
The encoder "understands" the input, the decoder generates the output.  
Encoder only models, such as BERT, produce contextual representations and then perform tasks such as sentiment analysis.  
Decoder only models, such as ChatGPT, generate text.  

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

## Additional Transformer References

### LLMS
- open weight facebook [llama](https://www.llama.com/)
- open source Andrej Karpathy [nanochat](https://github.com/karpathy/nanochat) [nanogpt](https://github.com/karpathy/nanoGPT)

### Vision Transformers (ViT) Papers
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [simCLR](https://arxiv.org/abs/2002.05709)
- [MAE Masked autoencoders ](https://arxiv.org/abs/2111.06377)
- [DINO](https://arxiv.org/abs/2104.14294)
- [iBOT](https://arxiv.org/abs/2111.07832)

