# RAG: Retrieval Augmented Generation
**RAG (retrieval augmented generation)** supplies an [LLM](llm.md) with text retrieved from an external source [1](#references). RAG addresses limitations such as knowledge cutoff date, proprietary or private data absent from pretraining, and hallucination on facts the model half-remembers. Rather than retraining the model to know something, RAG puts the relevant text in the prompt and lets the model read it.

RAG has three stages:
1. **Retrieve** - search a knowledge base for the chunks most relevant to the user's query.
2. **Augment** - insert the retrieved chunks into the prompt alongside the query.
3. **Generate** - the LLM answers grounded in the retrieved text, often citing its sources.

### Knowledge base
The **knowledge base** is the document collection RAG retrieves from: internal wikis, PDFs, codebases, past support tickets — any text. It is indexed ahead of time (offline) so retrieval is fast at query time.

Indexing pipeline: documents → chunks → embeddings → vector database.

**Chunks** are the units of retrieval. Documents are split into passages because embedding whole documents blurs their meaning into one vector, and because retrieved text must fit in the prompt. Chunking strategies range from fixed-size token windows (often with overlap so sentences aren't cut mid-thought) to structure-aware splitting on paragraphs, sections, or markdown headers.

**Embeddings** are vectors that represent each chunk's meaning, produced by an [embedding model](transformer.md) (typically an encoder trained so that semantically similar text maps to nearby vectors). Each chunk's embedding is stored in a **vector database** (e.g. FAISS, Chroma, pgvector, Pinecone). At query time the question is embedded with the same model, and the database returns the chunks whose vectors are closest — nearest neighbors by [cosine similarity](la.md#cosine-similarity) or dot product.

#### Hyperparameters
- **Chunk size** (in tokens) - small chunks (~100-300 tokens) give precise matches but lose surrounding context; large chunks (~500-1000) preserve context but dilute the embedding and spend more of the prompt. Chunk **overlap** is a related setting.
- **Embedding size** (vector dimension) - set by the choice of embedding model (commonly 384-3072). Larger embeddings capture more nuance but cost more storage and compute per similarity comparison.
- **top-k** - how many chunks to retrieve and insert into the prompt.

### The retriever
The **retriever** is the search component that maps a query to the most relevant chunks. The two main search families:

- **Keyword search** (lexical search) matches the literal words in the query, ranked by algorithms such as **BM25**. Strong on exact terms: names, part numbers, error codes, jargon. Fails on synonyms — "car" doesn't match "automobile."
- **Semantic search** (vector search) matches meaning via embedding similarity, so "car" matches "automobile." Fails where exact strings matter — rare proper nouns or identifiers the embedding model doesn't represent well.

**Hybrid search** runs both and merges the ranked lists (commonly with reciprocal rank fusion), getting the exact-match strength of keywords and the paraphrase strength of embeddings. Hybrid is the standard production choice.

A **reranker** is an optional second stage: a model that takes the query paired with each candidate chunk and scores relevance directly. It is more accurate than embedding similarity (it reads query and chunk together rather than comparing vectors computed separately) but too slow to run over the whole knowledge base — so the retriever fetches a broad candidate set (say 50) and the reranker picks the best few.

### Evaluation
RAG quality is measured at two levels: did the retriever find the right chunks (**recall@k**, precision), and did the model answer faithfully from them (**groundedness** — answers supported by retrieved text rather than hallucinated). A RAG system can fail at either stage independently, so they are diagnosed separately.

### RAG vs alternatives
- **Fine-tuning** teaches the model style and skills but is a poor fit for facts that change or must be attributable; RAG updates by re-indexing documents, no training required.
- **Long context** - stuffing all documents into the prompt each query becomes possible as context windows grow, but retrieval remains cheaper at scale and focuses the model on relevant text.
- RAG is a single-shot pipeline; [agents](llm.md#agents) generalize it — an agent can search, read results, refine its query, and search again in a loop (**agentic retrieval**).


<img width="2628" height="1476" alt="unnamed" src="https://github.com/user-attachments/assets/ed75110f-d550-416a-b594-0f269f8cddb8" />

## Class
[Deeplearning.ai](https://www.deeplearning.ai/courses/retrieval-augmented-generation)

## References
1. 2020 RAG paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. 2020 Dense passage retrieval paper [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

