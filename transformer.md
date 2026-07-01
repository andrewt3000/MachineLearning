# Transformers
A Transformer is a neural network architecture built with attention layers. It was introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) [blog](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/). Transformers are the large language models in multimodal foundation models such as Google Gemini, Open AI Chat GPT, Anthropic Claude.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/transformer.png" height='460px' width='326px'/>  
<sub>Transformer from <a href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a> (Vaswani et al., 2017)</sub>

The original “attention is all you need” paper was for language translation. 
The attention paper model has an encoder on the left and a decoder on the right.
The encoder "understands" the input language, the decoder generates the output language.  
Both encoder and decoder contain “Multi-head attention” MHA layers. [pytorch implementation](https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html) 

### Input Representation
- The model receives a sequence of tokens (words/subwords), each mapped to a vector embedding.
- A positional encoding is added to each token so the model knows the order.
- Together, these embeddings form a matrix where each row is a token vector.

### Self-Attention
Self-attention computes how much each token should “pay attention” to every other token in the sequence.
To do this, the model learns 3 projection matrices:

These transform each token vector into:
- Q (Query) – represents the current token.
- K (Key) – represents the tokens being compared against.
- V (Value) – carries the actual information to mix together.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Multi-Head Attention
Instead of one attention operation, the model uses h attention heads:
- Each head learns different relationships (syntax, long-range dependencies, semantics, etc.).
- Heads are concatenated and combined into a single output.

This lets the model analyze many patterns in parallel.

### Training Tricks
Label Smoothing is a regularization technique.  

## Bert
[Bert](https://arxiv.org/abs/1810.04805) is an encoder only model. It produce contextual representations and then perform tasks such as sentiment analysis.  
Encoder only models were popular from 2018 - 2022.  

## LLMs Large Language Models
LLMs became popular starting in 2022 with the success of ChatGPT.  
LLMs are language models in that they predict the next word.  
They are large in terms of parameter count, and tokens they are trained on. They also require large amount of compute resrouces to create frontier models.
The architecture of an llm is typically an autoregressive decoder only tranformer models.  
**Mixture of experts (MoE)** is an llm architecture that places a learned gate that routes to the "expert" network that can best handle the token. Typically, a sparse MoE will only use a fraction (top-1 or top-2 routing) of the parameters of the model, increasing efficiency.    
**Temperature** is a hyperparameter that controls the randomness and creativity of the model's generated text. 

## Academic history
- 2017 transformer paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) [blog](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/)
- 2018 bert paper, encoder only [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob](https://arxiv.org/abs/1810.04805)
- 2019 T5 paper, encoder-decoder, text to text [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- 2020 GPT3 autoregressive language model [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- 2020 Scaling laws paper [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### LLMS
- Google [gemini](https://gemini.google.com/), Open AI [Chat GPT](https://chatgpt.com/), Anthropic [Claude](https://claude.ai/), xAI [grok](https://grok.com/), [perplexity](https://www.perplexity.ai/), [mistral](https://chat.mistral.ai/chat), 
- open weight Facebook [llama](https://www.llama.com/), [deepseek v3](https://github.com/deepseek-ai/deepseek-v3), Alibaba [Qwen](https://qwen.ai/home)  (See [models on hugging face](https://huggingface.co/models) for up to date directory)  
- open source Andrej Karpathy [nanochat](https://github.com/karpathy/nanochat) [nanogpt](https://github.com/karpathy/nanoGPT)

### Vision Transformers (ViT) Papers
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [simCLR](https://arxiv.org/abs/2002.05709)
- [MAE Masked autoencoders ](https://arxiv.org/abs/2111.06377)
- [DINO](https://arxiv.org/abs/2104.14294)
- [iBOT](https://arxiv.org/abs/2111.07832)

### Tutorials / classes
[Stanford CME 295](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy) - Tranformers & Large Language Models
