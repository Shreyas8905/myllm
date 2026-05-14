# MyLLM

## Abstract

**MyLLM** is a high-efficiency decoder-only transformer model optimized for autoregressive sequence modeling. The design integrates modern architectural refinements including **Grouped-Query Attention (GQA)**, **Rotary Positional Embeddings (RoPE)**, and **SwiGLU-style Feed-Forward Networks (FFN)**. By employing a pre-normalization strategy and bias-free linear projections, MyLLM achieves enhanced gradient stability and parameter efficiency. This document presents the mathematical foundations, architectural justifications, and practical training methodologies utilized in the development of the model.

## 1. Architectural Foundation: Causal Modeling

MyLLM is structured as a pure decoder-only transformer, a design choice predicated on the requirements of causal language modeling where the objective is to model the joint probability of a sequence as $P(x) = \prod_{t=1}^T P(x_t | x_{<t})$.

### 1.1 Pre-Normalization and Residual Connectivity

The architecture utilizes a **Pre-Norm** configuration, where normalization is applied to the input of each sub-layer prior to the attention or feed-forward operations.

- **Mathematical Flow**: Given an input $x_l$ to layer $l$, the output $x_{l+1}$ is computed as:
  $x_{l+1} = x_l + \text{Sublayer}(\text{RMSNorm}(x_l))$
- **Stability Justification**: Unlike the Post-Norm structure introduced in the original Transformer, Pre-Norm architectures maintain a clean identity path for the residual flow. As demonstrated by **Xiong et al. (2020)** in _"On Layer Normalization in the Transformer Architecture,"_ this configuration prevents the vanishing or exploding gradient issues common in deep stacks, enabling more stable training for the model's 12-layer depth.

### 1.2 Root Mean Square Layer Normalization (RMSNorm)

MyLLM adopts **RMSNorm** as its primary normalization mechanism.

- **Mathematical Definition**: RMSNorm regularizes the summed inputs to a neuron according to the root mean square:
  $\bar{a}_i = \frac{a_i}{\sqrt{\frac{1}{n} \sum_{j=1}^n a_j^2 + \epsilon}} \cdot g_i$
- **Practical Utility**: By removing the mean-centering and additive bias terms found in standard LayerNorm, RMSNorm reduces computational overhead while preserving the re-scaling invariance that stabilizes training. This implementation uses a small $\epsilon$ of $1e-6$ for numerical stability.

## 2. Attention Mechanism: Grouped-Query Attention (GQA)

To address the memory bandwidth bottlenecks associated with the KV (Key-Value) cache during inference, MyLLM utilizes **Grouped-Query Attention**.

- **Mechanism**: The model partitions its 9 query heads into 3 groups, with each group sharing a single set of KV heads ($n\_heads=9, n\_kv\_heads=3$).
- **Practical Advantage**: GQA serves as a middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). As detailed by **Ainslie et al. (2023)** in _"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints,"_ this reduces the memory footprint of the KV cache during autoregressive generation while maintaining significantly higher modeling capacity than pure MQA.

---

## 3. Positional Representation: Rotary Positional Embeddings (RoPE)

In lieu of traditional absolute learned positional embeddings, MyLLM implements **Rotary Positional Embeddings (RoPE)**.

- **Mathematical Operation**: RoPE encodes position by rotating the query ($q$) and key ($k$) vectors in a complex plane. For a position $m$ and dimension pair $d$, the rotation is defined as:
  $\begin{pmatrix} q_1 \\ q_2 \end{pmatrix} \mapsto \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}$
- **Theoretical Merit**: Following **Su et al. (2021)** in _"RoFormer: Enhanced Transformer with Rotary Position Embedding,"_ RoPE allows the model to capture relative distances between tokens naturally. This approach avoids the constraints of a fixed sequence length lookup table and generally facilitates better extrapolation to sequences longer than the 128-token training window.

## 4. Feed-Forward Network: SwiGLU Gated Linear Unit

The Feed-Forward Network (FFN) block is implemented using a gated design with an intermediate expansion dimension of 1,536.

- **Formulation**: The output of the FFN is computed using the **SwiGLU** activation:
  $\text{FFN}(x) = (\text{SiLU}(xW_{gate}) \cdot xW_{up})W_{down}$
- **Architectural Contrast**: Standard Transformers utilized a two-layer ReLU-based MLP. MyLLM’s implementation follows the design of **Llama**, utilizing a gated mechanism that has been shown to offer superior expressivity and convergence properties per floating-point operation.

## 5. Training Methodology and Implementation

The model was implemented using the **Keras/TensorFlow** ecosystem with a focus on modularity and streaming data efficiency.

### 5.1 Dataset Pipeline

Training utilized the `cosmopedia-v2` subset of the `HuggingFaceTB/smollm-corpus`. To facilitate training on large-scale data without substantial local storage, a **streaming dataset pipeline** was implemented via `load_dataset(..., streaming=True)`. This pipeline extracts `seq_len + 1` tokens, which are then shifted to create input-label pairs for next-token prediction.

### 5.2 Optimization Suite

The model was optimized using the **AdamW** algorithm, which decouples weight decay from the gradient update to improve regularization.

- **Hyperparameters**: A learning rate of $3e-4$ and weight decay of $0.01$ were applied over 50 epochs.
- **Loss Function**: Sparse Categorical Cross-Entropy was used to minimize the negative log-likelihood of the target tokens.

## 6. Getting Familiar with Code

### 6.1 What This Repository Contains

- `config.py` defines the model and training hyperparameters in a single typed configuration object.
- `data/dataset.py` builds a streaming `tf.data.Dataset` from the Hugging Face `smollm-corpus` dataset and applies tokenization.
- `models/layers.py` implements the reusable transformer components: `RMSNorm`, rotary positional embedding logic, `GQA`, and `MLP`.
- `models/myllm.py` assembles the decoder blocks and exposes `build_myllm()`.
- `train.py` wires the dataset, model, optimizer, loss, and checkpoints into a training run.

### 6.2 Project Layout

```text
myllm/
├── config.py
├── data/
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── layers.py
│   └── myllm.py
├── train.py
├── requirements.txt
├── checkpoints/
└── readme.md
```

### 6.3 Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py
```

The first run will download the tokenizer and stream the dataset from Hugging Face.

### 6.4 Notes On Current Limitations

- The dataset is intentionally truncated to 10,000 streamed samples, so this is a prototype-scale training run rather than a full pretraining recipe.
- There is no explicit validation split in the current code.
- There is no text generation script yet.
- The implementation assumes `seq_len` divides the training shapes cleanly and does not include padding-aware loss masking beyond the dataset truncation strategy.
- The model does not currently use weight tying between embeddings and the LM head.

### 6.5 Future Improvements

- Add a validation dataset and perplexity reporting.
- Add autoregressive text generation with temperature, top-k, and top-p sampling.
- Add checkpoint resume support.
- Add mixed-precision training for better throughput on supported hardware.
- Consider weight tying between the token embedding table and the output head.
- Add a small benchmark or smoke test to verify the model compiles and trains on dummy data.

## 7. References

- **Ainslie, J., et al. (2023).** _GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints._ https://aclanthology.org/2023.emnlp-main.298.pdf https://github.com/google/flaxformer
- **Loshchilov, I., & Hutter, F. (2017).** _Decoupled Weight Decay Regularization._ https://arxiv.org/pdf/1711.05101 https://github.com/loshchil/AdamW-and-SGDW
- **Su, J., et al. (2021).** _RoFormer: Enhanced Transformer with Rotary Position Embedding._ https://arxiv.org/pdf/2104.09864 https://github.com/ZhuiyiTechnology/roformer
- **Touvron, H., et al. (2023).** _LLaMA: Open and Efficient Foundation Language Models._ https://formacion.actuarios.org/wp-content/uploads/2024/05/2302.13971-LLama-Open-and-Efficient-Foundation-Language-Models.pdf https://github.com/meta-llama/llama
- **Vaswani, A., et al. (2017).** _Attention is All You Need._ https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf https://github.com/tensorflow/tensor2tensor
- **Xiong, R., et al. (2020).** _On Layer Normalization in the Transformer Architecture._ https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
- **Zhang, B., & Sennrich, R. (2019).** _Root Mean Square Layer Normalization._ https://proceedings.neurips.cc/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf https://github.com/bzhangGo/rmsnorm
