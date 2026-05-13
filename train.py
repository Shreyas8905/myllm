# train.py
import os
import keras
import tensorflow as tf
from config import MyllmConfig
from models.myllm import build_myllm
from data.dataset import get_dataset

def configure_device():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected by TensorFlow. Training will run on CPU.")
        return "CPU"

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gpu_names = ", ".join(gpu.name for gpu in gpus)
    print(f"TensorFlow detected GPU(s): {gpu_names}")
    return "GPU"

def run_training():
    device = configure_device()
    config = MyllmConfig()
    print(f"Initializing MyLLM Training Pipeline...")

    dataset, tokenizer = get_dataset(config)
    print("Streaming dataset initialized. Tokenizer loaded.")

    model = build_myllm(config)
    
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    checkpoint_path = "checkpoints/myllm_120m_weights"
    os.makedirs("checkpoints", exist_ok=True)
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
        verbose=1
    )

    print(f"Starting training for {config.epochs} epoch(s) on {device}...")
    model.fit(
        dataset,
        epochs=config.epochs,
        callbacks=[checkpoint_callback]
    )

    final_model_path = "models/myllm_120m_final_weights"
    os.makedirs("models", exist_ok=True)
    model.save_weights(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")

if __name__ == "__main__":
    run_training()
