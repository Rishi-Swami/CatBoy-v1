# catboy.py
import torch
from tokenizer import SimpleTokenizer
from model import GPTModel
from generate import generate

MODEL_PATH = "saved_model_learning/gpt_chatbot_final.pth"
TOKENIZER_PATH = "saved_model_learning/tokenizer.json"

def interactive_chat(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH, max_new_tokens=40):
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)

    model = GPTModel(vocab_size=len(tokenizer.vocab), max_position_embeddings=32)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    print("CatBoy is ready to chat! Type 'bye' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("bye", "exit", "quit"):
            print("CatBoy: Bye!")
            break
        prompt = user_input
        response = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.9, top_k=40)
        print("CatBoy:", response)

if __name__ == "__main__":
    interactive_chat()
