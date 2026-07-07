# Transformers and LLMs
A **Transformer** is a neural network architecture that processes sequences and consists of two sub-layers: multi-head attention and a position-wise feed-forward network (FFN), each wrapped with a residual connection and layer normalization. Tranformers have no recurrence and largely replaced [recurrent neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md) such as  grus and lstms for processing sequences, such as text. Transformers are the primary architecture in large language models and vision transformers.  
 
### Attention is all you need
Transformers were introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) [blog](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/). The original “attention is all you need” paper is for language translation. It is a supervised learning task trained on a parallel corpus of data in different languages.

The attention paper model has an encoder on the left that "understands" the input language and a decoder on the right that generates the output language.  Both encoder and decoder contain attention blocks.

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/transformer.png" height='460px' width='326px'/>  
<sub>Transformer from <a href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a> (Vaswani et al., 2017)</sub>

## Attention Layers

### Input Representation
In text based transformers, words are tokenized (including an end of sequence token), then tokens are mapped to embeddings, and positional information is added. The positional encoding maintains the information about the position of the token that would otherwise be lost in the permution invariant attention block. Originally attention is all you need used sinusoidal positional encodings. However, modern llms use positional encoding schemes primarily **RoPe** (Rotary Position Embedding), and to a lesser extent **ALiBi** (Attention with Linear Biases).  

Externally, the transformer appears to accept a variable length input of tokens. Internally, transformer's input is a matrix with maximum number of input embeddings. The program masks out the empty slots in the matrix with padding tokens. Advanced implementations eliminate empty slots. The maximum number of tokens is referred to by many different names such as context window, context length, context size, attention window, and token size.   

### Self Attention
Intuitively, self attention computes how much each token in the matrix should “pay attention” to every other token in the sequence. Each token can attend to every token in the sequence. 
In practice, attention layers learns 3 projection matrices. The 3 matrices borrow an analogy from information retrieval and use the names query, key and value.  

These transform each token vector into:
- Q (Query) – represents what this token is looking for. Q = X $\bullet$ W<sub>Q</sub>
- K (Key) – represents what each token offers to be matched against. K = X $\bullet$ W<sub>K</sub>
- V (Value) – represents the content retrieved. V = X $\bullet$ W<sub>V</sub>

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Cross attention
**Self attention** has a single source input X. **Cross attention** merges 2 input sources. Q comes from the target sentence matrix ($Y$), while $K$ and $V$ come from the source sentence matrix ($X$).  
-  Q = Y $\bullet$ W<sub>Q</sub>
-  K = X $\bullet$ W<sub>K</sub>
-  V = X $\bullet$ W<sub>V</sub>

### Multi-Head Attention
The number of heads (h) is a hyperparameter.  
Intuitively, each head learns different relationships (syntax, long-range dependencies, semantics, etc.).
In practice, each head has its own set of learnable projection matrices ($W_Q, W_K, W_V$). So multiple heads allows the model to analyze many patterns in parallel.
Heads are concatenated and combined into a single output.


### Transformer block
- In attention is all you need, **layer normalization** is added after each sub-layer with residual connections. In the original paper Layer norm is post; nearly all modern LLMs use pre-LN. Layer norm stabilizes and accelerates training by mitigating internal covariate shift and exploding/vanishing gradient issues.  
- **label Smoothing** is a regularization technique that prevents a model from becoming overly confident in its predictions.  
- **Attention map** shows how strongly tokens relate to one another at a specific layer and attention head.
- **Key-Value Cache** is a memory optimization technique for autoregressive text generation.  


### Pytorch implementation 
- [Multi-head attention layers](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) 
- [Layer Normalization layers](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)


## Bert
[Bert](https://arxiv.org/abs/1810.04805) is an encoder only model. Bert is trained with transfer learning. Bert is pretrained with proxy tasks, namely 2 objective functions, MLM (masked language modeling) and NSP (next sentence prediction). The pretraining creates a base model that is fine tuned on tasks such as classification like spam detection, and sentiment analysis.  
Encoder only models were dominant from 2018 - 2022.  

# LLMs Large Language Models
LLMs became popular starting in 2022 with the success of ChatGPT.  
LLMs are language models in that they predict the next token.  
LLMs are large in terms of parameter count, and the number of tokens they are trained on. They also require large amount of compute resources to create frontier models.

### LLM Architecture
The architecture of an llm is typically an **autoregressive** decoder only transformer model. Autoregressive means that the model uses its own outputs as inputs for the next step. LLMs are trained using self supervised learning on a large corpus of text. During training, the model predicts every next token in the corpus in parallel; a causal mask prevents each position from attending to future tokens. At inference, the predicted next token output is then fed to the input and the next token is predicted recursively until the model generates an end of sequence token.  

When the llm outputs the next token prediction rather than greedily choosing the next highest probability word it typically uses an algorithm to choose the next best sequence of words. Typically llms use top-k and top-p **sampling**. **Beam search** is a classic alternative for seq2seq tasks.  

**Temperature** is a hyperparameter that controls the randomness and creativity of the model's generated text. A high temperature has a more uniform output distribution and will be more random. A low temperature has a spiky distribution and has a more predictable output. 

**Mixture of experts (MoE)** is an llm architecture where the FFN in each block is replaced by multiple expert FFNs plus a router that sends each token to the top-k experts. 

### Prompting
A **System prompt** is instructions given to an llm before the user input. It sets the model's context, instructions and constraints.  

**Adversarial prompting** is the practice of intentionally crafting inputs to trick, manipulate, or test Large Language Models (LLMs) into behaving outside their intended parameters.   

**Chain of thought** (CoT) prompting is a technique that instructs large language models to "show their work" before giving an answer. By breaking complex problems down into small, logical steps, the AI mimics human reasoning.  

## Academic history
- 2017 transformer paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) [blog](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/)
- 2018 bert paper, encoder only [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- 2019 T5 paper, encoder-decoder, text to text [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- 2020 GPT3 autoregressive language model [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- 2020 Scaling laws paper [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### LLMS
- Google [Gemini](https://gemini.google.com/)
- Open AI [Chat GPT](https://chatgpt.com/)
- Anthropic [Claude](https://claude.ai/)
- xAI [grok](https://grok.com/)
- [Mistral](https://chat.mistral.ai/chat)

### Open Weight llms
- [Latest models on hugging face](https://huggingface.co/models)
- Meta [llama](https://www.llama.com/)
- [deepseek v3](https://github.com/deepseek-ai/deepseek-v3)
- Alibaba [Qwen](https://qwen.ai/home)
- z.ai [GLM 5.2](https://huggingface.co/zai-org/GLM-5.2) [paper](https://arxiv.org/abs/2602.15763)  


Toy, open source models by Andrej Karpathy [nanochat](https://github.com/karpathy/nanochat) [nanogpt](https://github.com/karpathy/nanoGPT)


### Tutorials / classes
[Stanford CME 295](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy) - Transformers & Large Language Models
