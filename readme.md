# MyLLM

MyLLM is a compact decoder-only language model implemented with Keras, TensorFlow, and Hugging Face tooling. The repository is intentionally small and explicit: the code shows the full path from raw text to tokenized training batches, through a transformer-style decoder stack, to a final language-model head.

The current implementation is optimized for clarity and experimentation rather than maximum throughput. It uses streaming data access, a custom attention block with grouped-query attention, rotary position embeddings, RMS normalization, and a SwiGLU-style feed-forward network.

## What This Repository Contains

- `config.py` defines the model and training hyperparameters in a single typed configuration object.
- `data/dataset.py` builds a streaming `tf.data.Dataset` from the Hugging Face `smollm-corpus` dataset and applies tokenization.
- `models/layers.py` implements the reusable transformer components: `RMSNorm`, rotary positional embedding logic, `GQA`, and `MLP`.
- `models/myllm.py` assembles the decoder blocks and exposes `build_myllm()`.
- `train.py` wires the dataset, model, optimizer, loss, and checkpoints into a training run.

## High-Level Architecture

MyLLM follows a standard decoder-only language modeling layout:

1. Token IDs are embedded into vectors.
2. The sequence passes through a stack of identical decoder blocks.
3. Each block applies pre-norm self-attention followed by a pre-norm feed-forward network, with residual connections around both sublayers.
4. A final RMSNorm prepares the hidden state for the vocabulary projection head.
5. The output head produces logits for next-token prediction at every position.

This is a causal language model, so each token can only attend to earlier tokens in the sequence.


## Why These Design Choices Were Made

### Decoder-only transformer

The model is a pure causal decoder because the task is next-token prediction. That keeps the architecture aligned with autoregressive generation: the model only needs to model $p(x_t \mid x_{<t})$. There is no encoder, cross-attention, or sequence-to-sequence overhead.

### Pre-norm residual blocks

Each sublayer is wrapped with `RMSNorm` before attention and before the feed-forward network. Pre-norm transformers are typically easier to optimize than post-norm variants because the residual path stays cleaner and gradients remain better behaved as depth increases. This is a practical choice for stable training, especially when the model is stacked to `num_layers = 12`.

### RMSNorm instead of LayerNorm

`RMSNorm` normalizes using only the root-mean-square of the activations and omits mean subtraction. That makes it simpler and slightly cheaper than LayerNorm while preserving the stabilization benefits that transformer blocks need. In a language model where every token passes through the same normalization path many times, reducing overhead here is worthwhile.

### Grouped-query attention

The attention block uses grouped-query attention (`n_heads = 9`, `n_kv_heads = 3`) rather than fully independent key/value heads for every query head. This reduces the size of the key/value projections and is especially useful when the model is scaled up or when attention memory becomes a bottleneck.

The main tradeoff is that some query heads share the same key/value heads. In practice, that is usually a good trade when you want a more efficient model without collapsing attention expressiveness too aggressively.

### Rotary position embeddings

The model uses RoPE instead of learned absolute position embeddings. This choice keeps position handling inside the attention computation and tends to generalize better to longer contexts than a fixed learned lookup table. It also avoids introducing another large position embedding matrix.

### SwiGLU-style feed-forward network

The MLP uses a gated design: `gate_proj`, `up_proj`, and `down_proj`. The gate is activated with SiLU and multiplied elementwise with the up projection, which is the SwiGLU pattern.

This is a strong default for LLMs because it is usually more expressive than a plain two-layer ReLU MLP at a similar compute budget. The wider `intermediate_dim = 1536` gives the block enough capacity to transform attention outputs into useful token representations.

### Bias-free dense layers

Most projection layers are created with `use_bias=False`. This is a common transformer convention that slightly reduces parameter count and keeps the block structure cleaner. In practice, biases are often not essential when normalization and residual connections are already doing the heavy lifting.

### Separate output head

The implementation uses a final dense layer named `lm_head` to project hidden states to vocabulary logits. The current code does not tie the input embedding matrix and output head weights. That keeps the implementation straightforward and easy to inspect. Weight tying could be added later if you want to reduce parameters and enforce symmetry between input and output token spaces.

### Streaming dataset pipeline

`data/dataset.py` uses `load_dataset(..., streaming=True)` so the corpus does not need to be fully materialized locally. That is a practical choice when prototyping on a large dataset. The generator takes only the first 10,000 streaming samples, which makes the current training loop lightweight and easy to run on limited hardware.

The dataset is tokenized to `seq_len + 1` tokens and then split into inputs and labels by shifting one position. That is the standard next-token training target for causal language models.

### AdamW optimizer

Training uses `AdamW`, which is a strong default for transformer models because it decouples weight decay from the adaptive learning-rate update. The configured `learning_rate` and `weight_decay` values are conservative and suitable for small-scale experimentation.

## Model Shapes and Flow

With the default configuration in `config.py`:

- `vocab_size = 50257`
- `d_model = 576`
- `n_heads = 9`
- `n_kv_heads = 3`
- `seq_len = 128`
- `intermediate_dim = 1536`
- `num_layers = 12`

The tensor flow is:

1. Input token IDs: `[batch, 128]`
2. Embeddings: `[batch, 128, 576]`
3. Each decoder block preserves hidden size: `[batch, 128, 576]`
4. Final LM head: `[batch, 128, 50257]`

The output is a vocabulary-sized logit vector for every sequence position.

## Training Pipeline

`train.py` performs the following steps:

1. Loads `MyllmConfig`.
2. Builds the streaming dataset and tokenizer.
3. Constructs the model with `build_myllm(config)`.
4. Configures `AdamW` and sparse categorical cross-entropy from logits.
5. Saves the best checkpoint to `checkpoints/myllm_120m.keras`.
6. Saves the final model to `models/myllm_120m_final.keras`.

This pipeline is intentionally minimal. It shows the essential training path without adding distributed training, mixed precision, or generation utilities.

## Project Layout

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

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py
```

The first run will download the tokenizer and stream the dataset from Hugging Face.

## Notes On Current Limitations

- The dataset is intentionally truncated to 10,000 streamed samples, so this is a prototype-scale training run rather than a full pretraining recipe.
- There is no explicit validation split in the current code.
- There is no text generation script yet.
- The implementation assumes `seq_len` divides the training shapes cleanly and does not include padding-aware loss masking beyond the dataset truncation strategy.
- The model does not currently use weight tying between embeddings and the LM head.

## Future Improvements

- Add a validation dataset and perplexity reporting.
- Add autoregressive text generation with temperature, top-k, and top-p sampling.
- Add checkpoint resume support.
- Add mixed-precision training for better throughput on supported hardware.
- Consider weight tying between the token embedding table and the output head.
- Add a small benchmark or smoke test to verify the model compiles and trains on dummy data.
