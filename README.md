# 🐾 CatBoy GPT - Minimal GPT-Style Chatbot 

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

A compact, educational implementation of a **transformer-based causal language model (mini-GPT)** built entirely from scratch in **PyTorch**.  
Created for learning, experimentation, and a deeper understanding of how GPT-style models work.

---

## ✨ Features

- 🧠 **Custom Regex Tokenizer** with save/load support  
- ⚙️ **Transformer Blocks** with causal masking  
- 🚀 **Training Script** (CPU/GPU compatible)  
- 💬 **Interactive Chat Program** (`catboy.py`)  
- 🎲 **Text Generation** with temperature & top-k sampling  

---

## ⚡ Quick Start

### 1️⃣ Install requirements

    pip install -r requirements.txt

### 2️⃣ Prepare dataset

Place your training text file in:

    dataset/training_data.txt

Each line should contain **one training example**.

### 3️⃣ Train the model

    python train.py

> 💡 Automatically uses GPU if available; otherwise defaults to CPU.

### 4️⃣ Chat interactively

    python catboy.py

### 5️⃣ Generate text from CLI

    python generate.py --prompt "Hello" --max_new_tokens 30

---

## 🧩 Project Structure

    CatBoy/
    ├── dataset/
    │   └── training_data.txt
    ├── saved_model_learning/   # auto-created, gitignored
    ├── tokenizer.py
    ├── model.py
    ├── utils.py
    ├── train.py
    ├── generate.py
    ├── catboy.py
    ├── README.md
    └── requirements.txt

---

## 🧠 Notes & Implementation Details

- `ignore_index` used in loss to skip learning padding tokens.  
- Transformer block includes **causal masking** and proper transposition for `nn.MultiheadAttention`.  
- Tokenizer supports `<PAD>` and `<UNK>` tokens, plus **save/load** to avoid train/infer mismatches.  
- Generation supports **temperature** and **top-k sampling**:
  - `temperature=0.9` - more creative  
  - `temperature=0.1` - more deterministic  

---

## ⚙️ Requirements

    torch
    tqdm

> 💡 On CPU-only systems: if you encounter CUDA errors, install the CPU-only PyTorch wheel as described in the official PyTorch docs: https://pytorch.org/get-started/locally/

---

## 🧼 Suggested `.gitignore`

    __pycache__/
    *.pyc
    saved_model_learning/
    *.pth
    tokenizer.json

---

## 🪶 License

This project is open-source and available under the **MIT License**.  
You’re free to use, modify, and distribute it with attribution.

---

## 💬 Final Notes

CatBoy GPT is designed as a **learning-focused, minimal GPT implementation** which is ideal for understanding the inner workings of modern transformer-based language models.

For larger-scale or production-grade models, consider:
CatBoy-v2 or CatBoy-v3
---

### Meow responsibly. Keep coding curiously.
