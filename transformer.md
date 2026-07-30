# Transformers and LLMs
A **Transformer** is a neural network architecture that processes sequences without recurrence, largely [replacing](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/) [recurrent neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/rnn.md) such as  GRUs and LSTMs. The transformer block architecture consists of two sub-layers: multi-head attention and a position-wise [feed-forward network](neuralNets.md) (FFN), each wrapped with a residual connection and layer normalization. Transformers are the primary architecture in large language models, vision transformers, automatic speech recognition systems such as whisper, and other state of the art machine learning domains and models.  
 
### Attention is all you need
Transformers were introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). The original “attention is all you need” paper implements language translation. It is a supervised learning task trained on a parallel corpus of data in different languages.

The attention paper model has an encoder on the left that "understands" the input language and a decoder on the right that generates the output language.  Both encoder and decoder contain attention blocks.

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/transformer.png" height='460px' width='326px'/>  
<sub>Transformer from <a href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a> (Vaswani et al., 2017)</sub>

## Attention Layers

### Input Representation
In text based transformers, words are tokenized (including an end of sequence token), then tokens are mapped to embeddings, and positional information is added. The positional encoding maintains the information about the position of the token that would otherwise be lost in the permutation invariant attention block. Originally attention is all you need used sinusoidal positional encodings. However, modern LLMs use positional encoding schemes primarily **RoPE** (Rotary Position Embedding), and to a lesser extent **ALiBi** (Attention with Linear Biases).  

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

### Multi-Head Attention
The number of heads (h) is a hyperparameter.  
Intuitively, each head learns different relationships (syntax, long-range dependencies, semantics, etc.).
In practice, each head has its own set of learnable projection matrices ($W_Q, W_K, W_V$). So multiple heads allows the model to analyze many patterns in parallel.
Heads are concatenated and combined into a single output.

### Attention Block
In attention is all you need, **layer normalization** is added after each sub-layer with residual connections. In the original paper Layer norm is post; nearly all modern LLMs use pre-LN. Layer norm stabilizes training, mitigates exploding/vanishing gradient issues, and permits training deep stacks. 

### Cross attention
**Self attention** has a single source input X. **Cross attention** merges 2 input sources. Q comes from the target sentence matrix ($Y$), while $K$ and $V$ come from the source sentence matrix ($X$).  
-  Q = Y $\bullet$ W<sub>Q</sub>
-  K = X $\bullet$ W<sub>K</sub>
-  V = X $\bullet$ W<sub>V</sub>


### Misc
- **Label smoothing** is a regularization technique that prevents a model from becoming overly confident in its predictions.  
- **Attention map** shows how strongly tokens relate to one another at a specific layer and attention head.
- **Key-Value Cache** is a memory optimization technique for autoregressive text generation.  


### Pytorch implementation 
[TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) implements an attention block as described in Attention is all you need. It defaults to layer norm last but can be changed with a parameter norm_first=True. You can also implement an attention block in more detail. 
[MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) implements a multi-head attention sublayer based on the attention is all you need paper.  Use  [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.linear.Linear.html) and [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) to fill out the attention block. Consider more advanced libraries for production code such as [flash-attn](https://github.com/dao-ailab/flash-attention).  

``` python
class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-Attention
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-Forward
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + self.dropout(attn_out)

        # Feed-Forward
        residual = x
        x = self.ln2(x)
        x = residual + self.ffn(x)
        
        return x

```

## BERT
[BERT](https://arxiv.org/abs/1810.04805) is an encoder only model. BERT is trained with transfer learning. BERT is pretrained with proxy tasks, namely 2 objective functions, MLM (masked language modeling) and NSP (next sentence prediction). The pretraining creates a base model that is fine tuned on tasks such as classification like spam detection, and sentiment analysis.  


## References
- 2017 transformer paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
- 2018 BERT paper, encoder only [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- 2019 T5 paper, encoder-decoder, text to text [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

### Tutorials / classes
- [Stanford CME 295 videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy) [syllabus](https://cme295.stanford.edu/syllabus/) - Transformers & Large Language Models
