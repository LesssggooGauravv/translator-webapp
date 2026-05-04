# 🌐 Seq2Seq Translator Web App

**English → Hindi** translation using an Attention-based Encoder-Decoder (Seq2Seq) model built with PyTorch, served via FastAPI, with a simple HTML/CSS/JS frontend.

## Architecture

Based on the **Attn-LSTM** architecture from the NNA course project (`nna_project_single_Script/`). The original `TemporalAttention` mechanism has been extended into **Bahdanau (Additive) Attention** for sequence-to-sequence translation.

## Project Structure

```
translator_webapp/
├── ml/                      # ML Model (PyTorch)
│   ├── model.py             # Encoder-Decoder + Bahdanau Attention
│   ├── dataset.py           # Data loading, tokenization, vocabulary
│   ├── train.py             # Training script
│   ├── translate.py         # Inference / translation
│   └── checkpoints/         # Saved weights (created after training)
│
├── backend/                 # FastAPI Backend
│   └── main.py              # API server with /translate endpoint
│
├── frontend/                # Frontend (HTML/CSS/JS)
│   ├── index.html           # Main page
│   ├── style.css            # Styling
│   └── app.js               # Fetch API logic
│
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup & Usage

### 1. Install Dependencies

```bash
cd translator_webapp
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m ml.train
```

This will:
- Download 5000 English-Hindi sentence pairs from HuggingFace
- Train the Seq2Seq model (~15 epochs)
- Save weights to `ml/checkpoints/`

### 3. Start the Backend Server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Open the Frontend

Open `frontend/index.html` in your web browser. Type English text and click "Translate"!

### 5. Test Translation via CLI

```bash
python -m ml.translate "hello how are you"
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model  | PyTorch (Seq2Seq + Bahdanau Attention) |
| Backend   | FastAPI + Uvicorn |
| Frontend  | HTML + CSS + Vanilla JS |
| Dataset   | IITB English-Hindi (HuggingFace) |

## Notes

- Translation quality depends on training data size and epochs
- For better results, train on a GPU with more data/epochs
- The model uses greedy decoding; beam search would improve quality
