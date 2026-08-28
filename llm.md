# LLMs Large Language Models
Large language models (LLMs) are text to text models that receive prompts and generate human-like responses. LLMs became popular starting in 2022 with the success of ChatGPT. LLMs are language models as they predict the next token. LLMs are large in terms of parameter count, the number of tokens for training, and the compute resources to create frontier models.  

### LLM Architecture
The architecture of an LLM is typically an **autoregressive** decoder only [transformer](transformer.md) model. Autoregressive means that the model uses its own outputs as inputs for the next step. At inference, the predicted next token output is then fed to the input and the next token is predicted recursively until the model generates an end of sequence token. LLMs are trained using self supervised learning on a large corpus of text. During training, the model predicts every next token in the corpus in parallel; a causal mask prevents each position from attending to future tokens.   

<img width="352" height="680" alt="llm architecture" src="https://github.com/user-attachments/assets/3cc524dd-dd88-481a-930d-b1214c2841bb" />


When the LLM outputs the next token prediction rather than greedily choosing the next highest probability word it typically uses an algorithm to choose the next best sequence of words. Typically LLMs use top-k and top-p **sampling**. **Beam search** is a classic alternative for seq2seq tasks.  

**Temperature** is a hyperparameter that controls the randomness and creativity of the model's generated text. A high temperature has a more uniform output distribution and will be more random. A low temperature has a spiky distribution and has a more predictable output. 

**Mixture of experts (MoE)** is an LLM architecture where the FFN in each block is replaced by multiple expert FFNs plus a router that sends each token to the top-k experts. 

### LLM Training Pipeline
LLMs are trained in stages. 
- **Pretraining** uses self-supervised next-token prediction on a massive text corpus, such as [fineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), to produce a base model with broad knowledge but no assistant behavior. 
- **SFT** (supervised fine-tuning) then trains on curated prompt → response demonstrations, teaching the base model to follow instructions and respond in an assistant format. Example dataset: [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- **Preference optimization** aligns the model with human preferences using ranked response pairs, either via **RLHF** (a learned reward model plus [reinforcement learning](rl.md), typically PPO) or **DPO** (a direct classification-style loss on preference pairs that skips the reward model and RL loop entirely).

**Chinchilla scaling law** showed that compute-optimal training uses roughly 20 tokens per parameter.  

**Knowledge cutoff date** is the final point in time covered by a Large Language Model’s (LLM) static training data.  

**FlashAttention** is a GPU IO-bandwidth optimization. It keeps the intermediate data of attention in fast on-chip SRAM rather than writing it out to slower HBM, eliminating the memory traffic that makes standard attention slow.  

### Prompting
A **System prompt** is instructions given to an LLM before the user input. It sets the model's context, instructions and constraints.  
Examples: [Claude System prompts](https://platform.claude.com/docs/en/release-notes/system-prompts)

**Adversarial prompting** is the practice of intentionally crafting inputs to trick, manipulate, or test Large Language Models (LLMs) into behaving outside their intended parameters.   

**Chain of thought** (CoT) prompting is a technique that instructs large language models to "show their work" before giving an answer. By breaking complex problems down into small, logical steps, the AI mimics human reasoning.  

**Reasoning models** train CoT behavior directly with RL rather than eliciting it via prompting. OpenAI's o1 (2024) was the first public example; DeepSeek-R1 (2025) was the first open-weight model with a published method (RL with verifiable rewards, RLVR).    

**GRPO** (Group Relative Policy Optimization), introduced in DeepSeekMath is an RL algorithm, a [PPO](rl.md#ppo) variant, that is currently widely used as an optimizer for reasoning training.  

[**RAG** (retrieval augmented generation)](rag.md) supplies an LLM with text retrieved from an external source. RAG addresses limitations such as knowledge cutoff date, proprietary or private data absent from pretraining, and hallucination on facts the model half-remembers.[2](#references)  

### Agents
An **agent** is an LLM that runs in a loop: it decides on an action, calls a **tool**, observes the result, and repeats until the task is done. Prompting and [RAG](rag.md) give the model information; agents give it the ability to *act* — run code, search the web, read and write files, query databases, or call APIs. The LLM is the reasoning engine; a surrounding program (the **agent harness**) executes the tool calls and feeds results back as new context.

**Tool use** (also called **function calling**) is the mechanism that makes agents possible. The tools are described to the model in the prompt (name, purpose, parameters — typically as a JSON schema). When the model decides a tool is needed, it emits a structured call rather than prose text; the harness executes it and returns the output as the next message. Tool use is trained into the model during SFT and preference optimization, not just prompted.  

The basic loop was formalized in **ReAct** (Reason + Act) [10](#references): the model alternates between a reasoning step ("I need to look up the current price") and an action step (call `get_price`), with the observation appended before the next reasoning step. Modern agents fold this into a single loop of chain of thought and tool calls.  

**MCP** (Model Context Protocol) is an open standard, [introduced by Anthropic](https://www.anthropic.com/news/model-context-protocol), for connecting LLMs to tools and data sources, so a tool written once (e.g. a database connector) can be used by any MCP-compatible model or harness.  

**Agentic coding** is currently the most successful application: tools such as Claude Code and Codex CLI give the model a terminal, a file system, and a test runner, and let it iterate on code autonomously — writing, running, reading errors, and fixing — over many steps.  

Distinctive challenges of agents relative to single-turn LLM use:
- **Compounding errors** - each step is a chance to go wrong, so a 95%-reliable step yields ~36% success over 20 steps. Reliability, not raw capability, is often the bottleneck.
- **Context management** - long tasks accumulate tool outputs that exhaust the context window; agents summarize, compress, or offload state to files and memory.
- **Prompt injection** - text a tool returns (a web page, an email) can contain instructions the model may mistakenly follow. This is the primary security concern for agents with access to real systems.
- **Evaluation** - benchmarks measure end-to-end task completion (e.g. SWE-bench for resolving real GitHub issues) rather than next-token accuracy.
- **Multi-agent systems** - an orchestrator delegates subtasks to specialized subagents (a researcher, a coder, a reviewer), each with its own context window.

## References
1. 2020 GPT3 autoregressive language model [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
2. 2020 RAG paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
3. 2020 Scaling laws paper [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
4. 2022 RLHF, InstructGPT paper [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
5. 2022 Chinchilla scaling law paper [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
6. 2022 CoT paper [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
7. 2022 Flash attention paper [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
8. 2024 DeepSeekMath paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
9. 2025 RLVR paper [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245)
10. 2022 ReAct paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
11. 2023 Toolformer paper [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

### LLMS
- Anthropic [Claude](https://claude.ai/)
- Open AI [Chat GPT](https://chatgpt.com/)
- Google [Gemini](https://gemini.google.com/)
- xAI [grok](https://grok.com/)
- [Mistral](https://chat.mistral.ai/chat)

### Open Weight LLMs
- [Latest models on hugging face](https://huggingface.co/models)
- Alibaba [Qwen](https://qwen.ai/home)
- [Kimi K3](https://www.kimi.com/)
- [deepseek v3](https://github.com/deepseek-ai/deepseek-v3)
- z.ai [GLM 5.2](https://huggingface.co/zai-org/GLM-5.2) [paper](https://arxiv.org/abs/2602.15763)  
- Meta [llama](https://www.llama.com/)

### Tutorials / classes
- [Stanford CME 295 videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy) [syllabus](https://cme295.stanford.edu/syllabus/) - Transformers & Large Language Models
- [Stanford CS336 Language Modeling from Scratch I 2025](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)
- [by hand ai](https://www.byhand.ai/)
- [RAG deeplearning.ai](https://www.deeplearning.ai/courses/retrieval-augmented-generation)
- [The Smol Training Playbook:
The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
- [nanogpt](https://github.com/karpathy/nanoGPT) Minimal GPT training in ~300 lines of PyTorch by Andrej Karpathy 
- [nanochat](https://github.com/karpathy/nanochat) Full ChatGPT clone training pipeline by Andrej Karpathy [tutorial](https://github.com/karpathy/build-nanogpt) and [video](https://youtu.be/l8pRSuU81PU)
