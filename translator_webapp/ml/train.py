"""
train.py - Training Script for Seq2Seq English-Hindi Translator

Usage:
    cd translator_webapp
    python -m ml.train

Trains the Encoder-Decoder model on a 5000-pair subset, saves:
    - ml/checkpoints/seq2seq_model.pth   (model weights)
    - ml/checkpoints/src_vocab.json      (source vocabulary)
    - ml/checkpoints/trg_vocab.json      (target vocabulary)
    - ml/checkpoints/config.json         (model hyperparameters)
"""

import os
import sys
import json
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure the project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.model import build_model
from ml.dataset import load_and_prepare_data, collate_fn, PAD_IDX


# =====================================================================
# HYPERPARAMETERS
# =====================================================================
CONFIG = {
    "num_samples": 5000,       # Number of sentence pairs from dataset
    "min_freq": 2,             # Minimum word frequency for vocab
    "max_len": 30,             # Max sentence length
    "embed_size": 256,         # Embedding dimension
    "hidden_size": 256,        # LSTM hidden size
    "num_layers": 1,           # Number of LSTM layers
    "dropout": 0.1,            # Dropout rate
    "batch_size": 64,          # Training batch size
    "epochs": 15,              # Number of training epochs
    "learning_rate": 1e-3,     # Adam learning rate
    "clip_value": 1.0,         # Gradient clipping value
    "teacher_forcing": 0.5,    # Teacher forcing ratio
    "checkpoint_dir": os.path.join(os.path.dirname(__file__), "checkpoints"),
}


def train_one_epoch(model, dataloader, optimizer, criterion, clip, device, tf_ratio):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    num_batches = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, trg, teacher_forcing_ratio=tf_ratio)

        # Reshape for loss: ignore <sos> token (index 0)
        # output: (batch, trg_len, vocab) -> (batch * (trg_len-1), vocab)
        # trg:    (batch, trg_len) -> (batch * (trg_len-1))
        output = output[:, 1:, :].contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping (from reference code)
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set. Returns average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        # No teacher forcing during evaluation
        output = model(src, trg, teacher_forcing_ratio=0.0)

        output = output[:, 1:, :].contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    print("=" * 60)
    print("  Seq2Seq Translator - Training Pipeline")
    print("=" * 60)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Step 1: Load Data ──
    print("\n--- Step 1: Loading Data ---")
    train_data, val_data, src_vocab, trg_vocab = load_and_prepare_data(
        num_samples=CONFIG["num_samples"],
        min_freq=CONFIG["min_freq"],
        max_len=CONFIG["max_len"],
    )

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ── Step 2: Build Model ──
    print("\n--- Step 2: Building Model ---")
    model = build_model(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        device=device,
        embed_size=CONFIG["embed_size"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # ── Step 3: Training Loop ──
    print("\n--- Step 3: Training ---")
    best_val_loss = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            CONFIG["clip_value"], device, CONFIG["teacher_forcing"]
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        # Perplexity (exp of loss)
        train_ppl = math.exp(min(train_loss, 100))
        val_ppl = math.exp(min(val_loss, 100))

        print(f"  Epoch {epoch:02d}/{CONFIG['epochs']:02d} | "
              f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, src_vocab, trg_vocab)
            print(f"    -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {CONFIG['checkpoint_dir']}")


def _save_checkpoint(model, src_vocab, trg_vocab):
    """Save model weights, vocabularies, and config."""
    ckpt_dir = CONFIG["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "seq2seq_model.pth"))

    # Save vocabularies
    src_vocab.save(os.path.join(ckpt_dir, "src_vocab.json"))
    trg_vocab.save(os.path.join(ckpt_dir, "trg_vocab.json"))

    # Save config for inference
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)


if __name__ == "__main__":
    main()
