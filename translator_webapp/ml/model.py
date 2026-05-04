"""
model.py - Seq2Seq Encoder-Decoder with Bahdanau Attention for Translation

Adapted from the Attn-LSTM in nna_project_single_Script/code-1.ipynb.
Original TemporalAttention: 2-layer MLP (Linear->Tanh->Linear) scoring encoder states.
Extended here to Bahdanau Attention conditioning on encoder + decoder state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """Bidirectional LSTM encoder. Reads the source (English) sentence."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        # Project bidirectional states -> single direction for decoder
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        # src: (batch, src_len) -> embedded: (batch, src_len, embed_size)
        embedded = self.dropout(self.embedding(src))
        # outputs: (batch, src_len, hidden*2), hidden/cell: (layers*2, batch, hidden)
        outputs, (hidden, cell) = self.lstm(embedded)
        # Combine forward[-2] and backward[-1] final hidden states
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden[-2], hidden[-1]], dim=1))).unsqueeze(0)
        cell = torch.tanh(self.fc_cell(torch.cat([cell[-2], cell[-1]], dim=1))).unsqueeze(0)
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention - extended from reference TemporalAttention.

    Original TemporalAttention scored only encoder outputs:
        attn = Sequential(Linear(H,H), Tanh(), Linear(H,1))
    
    Bahdanau adds decoder state conditioning:
        energy = tanh(W_enc @ enc_out + W_dec @ dec_hidden)
        score  = V @ energy -> softmax -> attention weights
        context = bmm(weights, enc_outputs)  [same trick as original]
    """

    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden_size, dec_hidden_size, bias=False)
        self.W_dec = nn.Linear(dec_hidden_size, dec_hidden_size, bias=False)
        self.V = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden), encoder_outputs: (batch, src_len, enc_hidden*2)
        enc_proj = self.W_enc(encoder_outputs)          # (batch, src_len, dec_hidden)
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)  # (batch, 1, dec_hidden)
        energy = torch.tanh(enc_proj + dec_proj)        # (batch, src_len, dec_hidden)
        scores = self.V(energy).squeeze(-1)             # (batch, src_len)
        attn_weights = F.softmax(scores, dim=-1)        # (batch, src_len)
        # Weighted sum via bmm (same as original TemporalAttention)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    """LSTM Decoder with Bahdanau Attention. Generates target (Hindi) tokens."""

    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size, 
                 num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(enc_hidden_size * 2, hidden_size)
        # LSTM input = embedding + context
        self.lstm = nn.LSTM(embed_size + enc_hidden_size * 2, hidden_size,
                            num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # Prediction from LSTM output + context + embedding
        self.fc_out = nn.Linear(hidden_size + enc_hidden_size * 2 + embed_size, vocab_size)

    def forward(self, target_token, decoder_hidden, decoder_cell, encoder_outputs):
        # target_token: (batch,) -> (batch, 1, embed_size)
        embedded = self.dropout(self.embedding(target_token.unsqueeze(1)))
        # Attention context
        context, attn_weights = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
        # LSTM input: [embedding; context]
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (decoder_hidden, decoder_cell))
        # Predict next token
        prediction = self.fc_out(torch.cat([
            lstm_output.squeeze(1), context, embedded.squeeze(1)
        ], dim=1))
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """Full Seq2Seq model wrapping Encoder + Decoder with teacher forcing."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        # Encode source
        encoder_outputs, hidden, cell = self.encoder(src)
        # First input is <sos> token
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = prediction
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs


def build_model(src_vocab_size, trg_vocab_size, device,
                embed_size=256, hidden_size=256, num_layers=1, dropout=0.1):
    """Factory function to create Seq2Seq model with given hyperparameters."""
    encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
    decoder = Decoder(trg_vocab_size, embed_size, hidden_size, hidden_size, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Weight init (adapted from reference _init_weights)
    def _init_weights(m):
        for name, param in m.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    model.apply(_init_weights)
    return model
