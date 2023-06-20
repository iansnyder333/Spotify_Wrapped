import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (batch, time) tensor of integers
        logits = self.token_embedding_table(idx)  # (batch, time, C)
        if targets is None:
            loss = None
        else:
            # loss function negative log loss, convert to 2D
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (b,t)
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus only on last time step
            logits = logits[:, -1, :]
            # Apply softmax for probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
