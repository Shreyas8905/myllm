from pydantic import BaseModel, Field

class MyllmConfig(BaseModel):
    vocab_size: int = Field(default=50257)
    d_model: int = Field(default=576)
    n_heads: int = Field(default=9)
    n_kv_heads: int = Field(default=3)
    seq_len: int = Field(default=128)
    intermediate_dim: int = Field(default=1536)
    num_layers: int = Field(default=12)
    batch_size: int = Field(default=8)
    learning_rate: float = Field(default=3e-4)
    weight_decay: float = Field(default=0.01)
    epochs: int = Field(default=50)
    repo_id: str = Field(default="Shreyas159/myLLM-120M")

if __name__ == "__main__":
    config = MyLLMConfig()
    print("Successfully initialized MyLLM Configuration!")
    print(f"Total Layers: {config.num_layers} | Embedding Dimension: {config.d_model}")