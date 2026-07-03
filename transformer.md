# Transformers
A **Transformer** is a neural network architecture that contain an **attention layer**.  Tranformers largely replaced rnns, grus and lstms for processing variable length sequences of inputs, such as text. Transformers are the primary architecture in large language models, vision transformers, and latent diffusion models.  

### Attention is all you need
Attention layers were introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) [blog](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/). The original “attention is all you need” paper was for language translation. It was a supervised learning task trained on parallel corpus of data in different languages.

The attention paper model has an encoder on the left that "understands" the input language and a decoder on the right that generates the output language.  Both encoder and decoder contain attention layers.

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/transformer.png" height='460px' width='326px'/>  
<sub>Transformer from <a href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a> (Vaswani et al., 2017)</sub>

## Attention Layers

### Input Representation
In text based transformers, words are tokenized, then encoded to embeddings, then a positional encoding is added and an end of sequence token is added. Externally, the transformer appears to accept a variable length input of tokens. Internally, transformer's input is a matrix with max number of input embeddings. The program masks out the empty slots in the matrix with padding tokens. Advanced implementations will eliminate empty slots.  

Originally attention is all you need used sinusoidal positional encodings. However, modern llms use positional encoding schemes such as **RoPe** (Rotary Position Embedding), and **ALiBi** (Attention with Linear Biases).  

### Self Attention
Intuitively, self attention computes how much each token in the matrix should “pay attention” to every other token in the sequence. Each token can attend to every token in the sequence. 
In practice, attention layers learns 3 projection matrices:

These transform each token vector into:
- Q (Query) – represents the current token. Q = X $\bullet$ W<sub>Q</sub>
- K (Key) – represents the tokens being compared against. K = X $\bullet$ W<sub>K</sub>
- V (Value) – carries the actual information to mix together. V = X $\bullet$ W<sub>V</sub>

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Cross attention
**Self attention** has a single source input X. **Cross attention** merges 2 input sources. Q comes from the target sentence matrix ($Y$), while $K$ and $V$ come from the source sentence matrix ($X$).  
-  Q = Y $\bullet$ W<sub>Q</sub>
-  K = X $\bullet$ W<sub>K</sub>
-  V = X $\bullet$ W<sub>V</sub>

### Multi-Head Attention
The number of heads (h) is a hyperparater.  
Intuitively, each head learns different relationships (syntax, long-range dependencies, semantics, etc.).
In practice, each head has its own set of learnable projection matrices ($W_Q, W_K, W_V$). So multiple heads allows the model analyze many patterns in parallel.
Heads are concatenated and combined into a single output.

- In attention is all you need, layer normalization is added after each network. Layer norm stabilizes and therefor accelerates training by mitigating internal xovariate shift and exploding and vanishing gradient issues.  

- **label Smoothing** is a regularization technique that prevents a model from becoming overly confident in its predictions.  

### Pytorch implmenation 
- [Multi-head attention layers](https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html) 
- [Layer Normalization layers](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html)


## Bert
[Bert](https://arxiv.org/abs/1810.04805) is an encoder only model. Bert is trained with transfer learning. Bert is pretrained with proxy tasks, namely 2 objective functions, MLM (masked language modeling) and NSP (next sentence prediction). The pretraining creates a base model that is fine tuned on tasks such as classification like spam detection, and sentiment analysis.  
Encoder only models were popular from 2018 - 2022.  

# LLMs Large Language Models
LLMs became popular starting in 2022 with the success of ChatGPT.  
LLMs are language models in that they predict the next word.  
LLMs are large in terms of parameter count, and the number of tokens they are trained on. They also require large amount of compute resrouces to create frontier models.

### LLM Architecture
The architecture of an llm is typically an autoregressive decoder only tranformer models.  
LLMs are trained using self supervised learning. They are trained on a large corpus and they mask the next word. 
 - **Mixture of experts (MoE)** is an llm architecture that places a learned gate that routes to the "expert" network that between the attention network and the fully connected networks. Typically, a sparse MoE will only use a fraction (top-1 or top-2 routing) of the parameters of the model, increasing efficiency.   
- When the llm outputs the word prediction rather than greedily choosing the next highest probability word it typically uses an algorithm to choose the next best sequence of words. Algorithms include  **beam search**, and top-k and top-p **sampling**.  
- **Temperature** is a hyperparameter that controls the randomness and creativity of the model's generated text. A high temperature has a more uniform output distribution and will be more random. A low temperature has a spikey distribution and has a more predictable output. 

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
