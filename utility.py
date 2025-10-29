# utils.py
import torch
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    """
    Basic dataset for causal LM training from plain lines.
    Each line is assumed to be a single text sequence.
    The dataset returns inputs and targets for next-token prediction:
    - input: [tok0, tok1, tok2, ..., tokN-1]
    - target:[tok1, tok2, ..., tokN]
    Sequences are padded/truncated to max_length.
    """

    def __init__(self, file_path, tokenizer, max_length=20):
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.max_length = max_length
        self.vocab = tokenizer.vocab
        self.data = []
        for line in lines:
            tokens = tokenizer.encode(line)
            if len(tokens) == 0:
                continue
            # truncate or pad
            if len(tokens) >= max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.vocab.get(tokenizer.pad_token, 0)] * (max_length - len(tokens))
            self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_tokens = tokens[:-1]  # all except last
        target_tokens = tokens[1:]  # all except first
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)

def get_dataloader(file_path, tokenizer, batch_size=4, max_length=20, shuffle=True):
    dataset = ChatDataset(file_path, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
