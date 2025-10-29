# generate.py
import torch
import argparse
from tokenizer import SimpleTokenizer
from model import GPTModel

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[-1]
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def generate(model, tokenizer, prompt, max_new_tokens=20, temperature=1.0, top_k=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokens = tokenizer.encode(prompt)
    pad_idx = tokenizer.vocab.get(tokenizer.pad_token, 0)

    for _ in range(max_new_tokens):
        context = tokens[-model.max_position_embeddings:] if hasattr(model, "max_position_embeddings") else tokens
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor)  # (1, seq_len, vocab)
        next_logits = logits[0, -1, :] / max(temperature, 1e-9)

        if top_k > 0:
            next_logits = top_k_logits(next_logits, top_k)

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(int(next_token))
        if next_token == pad_idx:
            break

    return tokenizer.decode(tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="saved_model_learning/gpt_chatbot_final.pth")
    parser.add_argument("--tokenizer", default="saved_model_learning/tokenizer.json")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    tokenizer = SimpleTokenizer()
    tokenizer.load(args.tokenizer)
    model = GPTModel(vocab_size=len(tokenizer.vocab))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    out = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k)
    print(">>>", out)
