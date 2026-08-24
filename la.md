# Linear Algebra for deep learning

- **scalar** - a single number, rank-0 tensor (magnitude only)
- **vector** - an ordered list of numbers, rank-1 tensor (magnitude and direction)
- **matrix** - table of values with rows and columns, rank-2 tensor

"**Rank**" in deep learning usually means the number of tensor axes (ndim); "**rank**" in linear algebra means the number of linearly independent rows/columns. LoRA's "low-rank" uses the linear algebra sense.  
By convention, vectors are lower case and matrices are upper case.  
### Dot Product
**dot product** or, more generally, the **inner product** takes two vectors of the same length and produces a single scalar measuring how aligned they are.

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

The identity a · b = |a| |b| cos(θ) connects the arithmetic to the angle θ between the vectors.

In ML, dot product is used in attention, linear layer matrix multiplication, embedding similarity search in [RAG](rag.md).

### Cosine Similarity
**cosine similarity** is the dot product of the normalized vectors. Dot product measures alignment and magnitude; cosine similarity measures alignment only

  cos_sim(a, b) = (a · b) / (|a| |b|) = cos(θ)

### Matrix Multiplication

**Matrix multiplication** (matmul) combines two matrices by taking dot products: each output entry (i, j) is the dot product of row i of A with column j of B.

```
C = AB
```

The inner dimensions must match: an (m×n) matrix times an (n×p) matrix produces an (m×p) matrix. Order matters — AB ≠ BA in general.

Two equivalent views:
- **Grid of dot products** - each output entry is a row·column dot product
- **Sum of outer products** - AB = Σ (column k of A) ⊗ (row k of B)

In ML, matmul is *the* core operation of deep learning: every linear layer computes y = Wx + b, attention computes QKᵀ, and nearly all training FLOPs are matrix multiplications. GPUs are fast at deep learning largely because they parallelize matmuls efficiently.

A linear layer's math is dot products; its implementation is a matmul — the batched form that hardware executes efficiently.

### Outer Product

The **outer product** takes two vectors and expands them into a matrix — the opposite of the inner product, which contracts them to a scalar.

```
a ⊗ b = abᵀ
```
A length-m vector and a length-n vector produce an m×n matrix. Example: [1, 2] ⊗ [3, 4, 5] = [[3, 4, 5], [6, 8, 10]].

The result is always a **rank-1 matrix** (linear algebra sense) — every row is a multiple of b, every column a multiple of a.

In ML:
- **Backprop**: the gradient of a linear layer's weights is an outer product: ∂L/∂W = (∂L/∂y) ⊗ x
- **LoRA**: a rank-r update BA is a sum of r outer products — why low-rank fine-tuning is so parameter-efficient
- **SVD**: any matrix decomposes into a sum of rank-1 outer products; truncating the sum gives low-rank compression

### Linear Algebra concepts
- **magnitude (norm)** - the length of a vector: |a| = √(a₁² + a₂² + ... + aₙ²)
- **unit vector** - vector with magnitude 1. Normalize any vector by dividing by its magnitude: a / |a|
- **parallel vectors** - point in the same (or opposite) direction; cross product is zero, cosine similarity is ±1
- **orthogonal (perpendicular) vectors** - dot product is zero; carry independent information
- **transpose** (Aᵀ) - flip rows and columns. Turns an m×n matrix into n×m. Used constantly, e.g. QKᵀ in attention
- **identity matrix** (I) - square matrix with 1s on the diagonal, 0s elsewhere. IA = A, the "multiply by 1" of matrices
- **inverse** (A⁻¹) - the matrix that undoes A: A⁻¹A = I. Only square, full-rank matrices have one. Rarely computed directly in ML (numerically unstable; solvers are used instead)
- **eigenvector / eigenvalue** - an eigenvector of A is a vector whose direction is unchanged by A, only scaled: Av = λv, where λ is the eigenvalue. Foundation of PCA and spectral methods
