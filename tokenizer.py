# tokenizer.py
import re
import json
from collections import Counter

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

    def build_vocab(self, texts, vocab_size=5000, add_special_tokens=True):
        """Build vocab from iterable of texts."""
        word_counts = Counter()
        for text in texts:
            word_counts.update(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))

        most_common = word_counts.most_common(vocab_size)
        self.vocab = {}
        start_idx = 1
        if add_special_tokens:
            # Reserve 0 for PAD
            self.vocab[self.pad_token] = 0
            self.vocab[self.unk_token] = 1
            start_idx = 2

        for idx, (word, _) in enumerate(most_common, start=start_idx):
            if word in self.vocab:
                continue
            self.vocab[word] = idx

        # reverse vocab
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}

    def encode(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return [self.vocab.get(tok, self.vocab.get(self.unk_token, 1)) for tok in tokens]

    def decode(self, token_ids, skip_pad=True):
        words = []
        for idx in token_ids:
            if skip_pad and idx == self.vocab.get(self.pad_token, 0):
                continue
            words.append(self.reverse_vocab.get(idx, self.unk_token))
        return " ".join(words)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {int(idx): word for word, idx in self.vocab.items()}
        # Note: when loading from JSON keys may be strings if saved differently.
        # Ensure reverse_vocab keys are ints.
        # If vocab keys are ints already, the comprehension above will still work.
