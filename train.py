# train.py
import os
import torch
import torch.optim as optim
from tqdm import tqdm

from tokenizer import SimpleTokenizer
from model import GPTModel
from utility import get_dataloader

def train(
    data_path="dataset/training_data.txt",
    save_dir="saved_model_learning",
    vocab_size=5000,
    max_length=32,
    batch_size=4,
    epochs=3,
    lr=3e-4,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Build tokenizer
    tokenizer = SimpleTokenizer()
    with open(data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    tokenizer.build_vocab(texts, vocab_size=vocab_size)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # Initialize model
    model = GPTModel(vocab_size=len(tokenizer.vocab), max_position_embeddings=max_length)
    model.to(device)

    dataloader = get_dataloader(data_path, tokenizer, batch_size=batch_size, max_length=max_length)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    pad_idx = tokenizer.vocab.get(tokenizer.pad_token, 0)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)  # shape (batch, seq_len-1)
            optimizer.zero_grad()
            logits = model(inputs)  # (batch, seq_len-1, vocab)
            # reshape for loss: (batch * seq_len, vocab) vs (batch * seq_len)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = loss_fn(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - avg loss: {avg_loss:.4f}")

        # checkpoint each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f"gpt_epoch{epoch+1}.pth"))

    # final save
    model_path = os.path.join(save_dir, "gpt_chatbot_final.pth")
    torch.save(model.state_dict(), model_path)
    print("Training complete. Model saved to:", model_path)

if __name__ == "__main__":
    train()
