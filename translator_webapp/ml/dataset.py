"""
dataset.py - English-Hindi Dataset Loader & Vocabulary Builder

Loads a small subset (5000 pairs) from the IITB English-Hindi parallel corpus
via HuggingFace datasets. Builds word-level vocabularies with special tokens.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# =====================================================================
# SPECIAL TOKENS
# =====================================================================
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# =====================================================================
# VOCABULARY CLASS
# =====================================================================

class Vocabulary:
    """Simple word-level vocabulary with special tokens."""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_count: Counter = Counter()

        # Add special tokens
        for idx, token in enumerate([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build(self, sentences: List[List[str]]):
        """Build vocabulary from tokenized sentences."""
        # Count all words
        for sentence in sentences:
            self.word_count.update(sentence)

        # Add words that appear >= min_freq times
        idx = len(self.word2idx)
        for word, count in self.word_count.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices, adding <sos> and <eos>."""
        indices = [SOS_IDX]
        for token in tokens:
            indices.append(self.word2idx.get(token, UNK_IDX))
        indices.append(EOS_IDX)
        return indices

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices back to tokens, stopping at <eos>."""
        tokens = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            tokens.append(self.idx2word.get(idx, UNK_TOKEN))
        return tokens

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        data = {"word2idx": self.word2idx, "min_freq": self.min_freq}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


# =====================================================================
# TOKENIZER (simple word-level)
# =====================================================================

def tokenize_en(text: str) -> List[str]:
    """Basic English tokenizer: lowercase + split on whitespace/punctuation."""
    text = text.lower().strip()
    text = re.sub(r"([.!?,])", r" \1 ", text)
    return text.split()


def tokenize_hi(text: str) -> List[str]:
    """Basic Hindi tokenizer: strip + split on whitespace/punctuation."""
    text = text.strip()
    text = re.sub(r"([।.!?,])", r" \1 ", text)
    return text.split()


# =====================================================================
# TRANSLATION DATASET
# =====================================================================

class TranslationDataset(Dataset):
    """PyTorch Dataset for English-Hindi translation pairs."""

    def __init__(self, src_encoded: List[List[int]], trg_encoded: List[List[int]]):
        self.src = [torch.tensor(s, dtype=torch.long) for s in src_encoded]
        self.trg = [torch.tensor(t, dtype=torch.long) for t in trg_encoded]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src[idx], self.trg[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, trg_padded


# =====================================================================
# DATA LOADING PIPELINE
# =====================================================================

def load_and_prepare_data(num_samples: int = 5000, min_freq: int = 2,
                          max_len: int = 30):
    """
    Load English-Hindi data, build vocabularies, and create encoded datasets.

    Uses HuggingFace 'cfilt/iitb-english-hindi' dataset (small subset).
    Falls back to a tiny dummy dataset if HuggingFace is unavailable.
    
    Args:
        num_samples: Number of sentence pairs to use
        min_freq: Minimum word frequency for vocabulary
        max_len: Maximum sentence length (longer sentences are filtered out)
    
    Returns:
        train_data, val_data: TranslationDataset objects
        src_vocab, trg_vocab: Vocabulary objects
    """
    en_sentences = []
    hi_sentences = []

    try:
        from datasets import load_dataset
        print("Loading IITB English-Hindi dataset from HuggingFace...")
        dataset = load_dataset("cfilt/iitb-english-hindi", split="train",
                               trust_remote_code=True)

        # Take a subset and filter by length
        count = 0
        for item in dataset:
            en_text = item["translation"]["en"]
            hi_text = item["translation"]["hi"]

            en_tokens = tokenize_en(en_text)
            hi_tokens = tokenize_hi(hi_text)

            # Filter: skip very long or very short sentences
            if 3 <= len(en_tokens) <= max_len and 3 <= len(hi_tokens) <= max_len:
                en_sentences.append(en_tokens)
                hi_sentences.append(hi_tokens)
                count += 1
                if count >= num_samples:
                    break

        print(f"Loaded {len(en_sentences)} sentence pairs.")

    except Exception as e:
        print(f"Could not load HuggingFace dataset: {e}")
        print("Using built-in dummy dataset for pipeline testing...")
        en_sentences, hi_sentences = _get_dummy_data()

    # Build vocabularies
    src_vocab = Vocabulary(min_freq=min_freq)
    trg_vocab = Vocabulary(min_freq=min_freq)
    src_vocab.build(en_sentences)
    trg_vocab.build(hi_sentences)
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(trg_vocab)}")

    # Encode all sentences
    src_encoded = [src_vocab.encode(s) for s in en_sentences]
    trg_encoded = [trg_vocab.encode(s) for s in hi_sentences]

    # Train/val split (90/10)
    split_idx = int(len(src_encoded) * 0.9)
    train_data = TranslationDataset(src_encoded[:split_idx], trg_encoded[:split_idx])
    val_data = TranslationDataset(src_encoded[split_idx:], trg_encoded[split_idx:])

    print(f"Train: {len(train_data)} pairs | Val: {len(val_data)} pairs")
    return train_data, val_data, src_vocab, trg_vocab


def _get_dummy_data():
    """Tiny hardcoded dataset for testing when HuggingFace is unavailable."""
    pairs = [
        ("hello how are you", "नमस्ते आप कैसे हैं"),
        ("what is your name", "आपका नाम क्या है"),
        ("i am a student", "मैं एक छात्र हूं"),
        ("this is a book", "यह एक किताब है"),
        ("where are you going", "आप कहां जा रहे हैं"),
        ("i like to read books", "मुझे किताबें पढ़ना पसंद है"),
        ("the weather is good today", "आज मौसम अच्छा है"),
        ("please give me water", "कृपया मुझे पानी दें"),
        ("i am learning hindi", "मैं हिंदी सीख रहा हूं"),
        ("good morning teacher", "सुप्रभात शिक्षक"),
        ("india is my country", "भारत मेरा देश है"),
        ("the sun rises in the east", "सूरज पूरब में उगता है"),
        ("she is playing in the garden", "वह बगीचे में खेल रही है"),
        ("we go to school every day", "हम हर दिन स्कूल जाते हैं"),
        ("he is eating food", "वह खाना खा रहा है"),
        ("the cat is sleeping", "बिल्ली सो रही है"),
        ("they are my friends", "वे मेरे दोस्त हैं"),
        ("this food is very tasty", "यह खाना बहुत स्वादिष्ट है"),
        ("i want to go home", "मैं घर जाना चाहता हूं"),
        ("the river is very long", "नदी बहुत लंबी है"),
    ]
    # Repeat to get more data points
    pairs = pairs * 50
    en_sentences = [tokenize_en(p[0]) for p in pairs]
    hi_sentences = [tokenize_hi(p[1]) for p in pairs]
    return en_sentences, hi_sentences
