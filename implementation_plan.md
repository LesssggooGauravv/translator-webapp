# Seq2Seq Translator Web App — Implementation Plan

Build a student-level full-stack English→Hindi translation web app based on your existing Attn-LSTM architecture from `nna_project_single_Script`.

## Reference Architecture (Read-Only)

From your `code-1.ipynb` Cell 1, the key pieces I'm adapting:

- **`TemporalAttention`** — 2-layer MLP (`Linear→Tanh→Linear`) that scores each encoder timestep, then softmax → weighted context vector via `bmm`.
- **`AttnLSTM`** — Multi-layer LSTM + TemporalAttention + Linear classifier.

I'll reshape this into a proper **Encoder-Decoder with Bahdanau Attention** for seq2seq translation.

---

## Proposed Changes

### Part 1 — ML Model (`translator_webapp/ml/`)

#### [NEW] [model.py](file:///d:/myFolder/sem6/nna/main/translator_webapp/ml/model.py)
- **Encoder**: Bidirectional LSTM that encodes the source (English) sentence into hidden states.
- **BahdanauAttention**: Adapted from your `TemporalAttention` — scores encoder outputs against the current decoder hidden state, softmax → context vector.
- **Decoder**: LSTM that takes `[embedding; context]` at each step, uses attention to look back at the encoder, predicts the next target (Hindi) token.
- **Seq2Seq**: Wraps Encoder + Decoder with teacher forcing support.

#### [NEW] [dataset.py](file:///d:/myFolder/sem6/nna/main/translator_webapp/ml/dataset.py)
- Loads a small English-Hindi parallel corpus using HuggingFace `datasets` (`cfilt/iitb-english-hindi`).
- Builds word-level vocabularies with `<pad>`, `<sos>`, `<eos>`, `<unk>` special tokens.
- Tokenizes and numericalized sentences; collates into padded batches.

#### [NEW] [train.py](file:///d:/myFolder/sem6/nna/main/translator_webapp/ml/train.py)
- Simple training loop: CrossEntropyLoss, Adam optimizer, gradient clipping.
- Saves model weights (`.pth`) + vocab objects (`.json`) after training.
- Uses a small subset (e.g., 5000 pairs) to keep training feasible on a student laptop.

#### [NEW] [translate.py](file:///d:/myFolder/sem6/nna/main/translator_webapp/ml/translate.py)
- Loads saved `.pth` weights and vocab.
- Tokenizes input English text → runs through encoder → greedy decodes from decoder → returns Hindi string.
- Exposed as a callable `translate(text: str) -> str` function.

---

### Part 2 — Backend (`translator_webapp/backend/`)

#### [NEW] [main.py](file:///d:/myFolder/sem6/nna/main/translator_webapp/backend/main.py)
- FastAPI app with CORS middleware (allows `localhost:*`).
- `POST /translate` endpoint: receives `{"text": "...", "src_lang": "en", "tgt_lang": "hi"}`, calls inference function, returns `{"translated_text": "..."}`.
- On startup, loads the model weights once into memory.
- Includes a `GET /health` endpoint for sanity checks.

#### [NEW] [requirements.txt](file:///d:/myFolder/sem6/nna/main/translator_webapp/backend/requirements.txt)
- `fastapi`, `uvicorn`, `torch`, `numpy`.

---

### Part 3 — Frontend (`translator_webapp/frontend/`)

#### [NEW] [index.html](file:///d:/myFolder/sem6/nna/main/translator_webapp/frontend/index.html)
- Clean, barebones Google Translate-style layout.
- Two text areas (source/target), language dropdowns, Translate button.
- Basic responsive design.

#### [NEW] [style.css](file:///d:/myFolder/sem6/nna/main/translator_webapp/frontend/style.css)
- Simple, clean CSS — light background, card-style container, basic button hover.
- No frameworks, no fancy animations — student-level.

#### [NEW] [app.js](file:///d:/myFolder/sem6/nna/main/translator_webapp/frontend/app.js)
- Fetch API to `POST` to backend.
- Loading state: button text changes to "Translating..." + disables button while waiting.
- Error handling with a basic alert/message.

---

### Root

#### [NEW] [README.md](file:///d:/myFolder/sem6/nna/main/translator_webapp/README.md)
- Project overview, file structure, setup instructions (pip install, train, run backend, open frontend).

---

## Open Questions

> [!IMPORTANT]
> **Dataset size**: I plan to use a small subset (~5,000 sentence pairs) from the IITB English-Hindi corpus for demo purposes. Training on a full dataset would take hours/days. Is this acceptable, or do you want a larger subset?

> [!NOTE]
> **Pre-trained weights**: Since this is a demo project, the model will produce *placeholder-quality* translations unless trained on significant data. The code structure and pipeline will be fully correct and functional. You can train it on Colab with a GPU for better results.

## Verification Plan

### Automated Tests
- Run `python -c "from ml.model import Seq2Seq; print('Model imports OK')"` to verify model code.
- Run `python -c "from backend.main import app; print('Backend imports OK')"` to verify FastAPI app.
- Start the FastAPI server and hit `/health` endpoint.
- Open `frontend/index.html` in browser and verify UI renders correctly.

### Manual Verification
- Visually inspect the frontend layout in the browser.
- Test the translate button loading state.
