import keras
import tensorflow as tf
from keras import layers
import numpy as np

keras.backend.set_floatx("float32")

class RMSNorm(layers.Layer):
    def __init__(self, dim: int, esp: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.esp = esp
        self.scale = self.add_weight(
            name="scale", 
            shape=(dim, ), 
            initializer="ones",
            trainable=True
        )

    def call(self, x):
        var = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(var + self.esp) * self.scale
    
def rope(q, k, seq_len: int, head_dim: int):
    inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype='float32') / head_dim))
    t = np.arange(seq_len, dtype='float32')
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = tf.cast(np.cos(emb), dtype=tf.float32)[None, :, None, :]
    sin = tf.cast(np.sin(emb), dtype=tf.float32)[None, :, None, :]

    def rotate_half(x):
        half_size = head_dim // 2
        x1 = x[..., :half_size]
        x2 = x[..., half_size:]
        return tf.concat([-x2, x1], axis=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MLP(layers.Layer):
    def __init__(self, d_model, intermediate_dim, **kwargs):
        super().__init__(**kwargs)
        self.gate_proj = layers.Dense(intermediate_dim, use_bias=False)
        self.up_proj = layers.Dense(intermediate_dim, use_bias=False)
        self.down_proj = layers.Dense(d_model, use_bias=False)

    def build(self, input_shape):
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)
        self.down_proj.build((input_shape[0], input_shape[1], self.gate_proj.units))
        self.built = True

    def call(self, x):
        gate = tf.nn.silu(self.gate_proj(x))
        activated = gate * self.up_proj(x)
        return self.down_proj(activated)

class GQA(layers.Layer):
    def __init__(self, d_model, n_heads, n_kv_heads, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.seq_len = seq_len
        self.num_queries_per_kv = n_heads // n_kv_heads

        self.q_proj = layers.Dense(n_heads * self.head_dim, use_bias=False)
        self.k_proj = layers.Dense(n_kv_heads * self.head_dim, use_bias=False)
        self.v_proj = layers.Dense(n_kv_heads * self.head_dim, use_bias=False)
        self.o_proj = layers.Dense(d_model, use_bias=False)

    def build(self, input_shape):
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_proj.build((input_shape[0], self.seq_len, self.n_heads * self.head_dim))
        self.built = True

    def call(self, x):
        batch_size = tf.shape(x)[0]
        q = tf.reshape(self.q_proj(x), (batch_size, self.seq_len, self.n_heads, self.head_dim))
        k = tf.reshape(self.k_proj(x), (batch_size, self.seq_len, self.n_kv_heads, self.head_dim))
        v = tf.reshape(self.v_proj(x), (batch_size, self.seq_len, self.n_kv_heads, self.head_dim))
        q, k = rope(q, k, self.seq_len, self.head_dim)

        if self.num_queries_per_kv > 1:
            k = tf.repeat(k, self.num_queries_per_kv, axis=2)
            v = tf.repeat(v, self.num_queries_per_kv, axis=2)

        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        mask = 1.0 - tf.linalg.band_part(tf.ones((self.seq_len, self.seq_len), dtype=tf.float32), -1, 0)

        scores = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
        scores = scores / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = scores + (mask[None, None, :, :] * -1e9)

        att_probs = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(att_probs, v)
        output = tf.transpose(output, (0, 2, 1, 3))
        output = tf.reshape(output, (batch_size, self.seq_len, -1))

        return self.o_proj(output)
