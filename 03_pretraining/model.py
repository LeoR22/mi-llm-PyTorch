import torch
import torch.nn as nn
from torch.nn import functional as F


class HeadAtencion(nn.Module):
    """Una sola cabeza de autoatención causal (enmascarada)."""
    def __init__(self, d_model, block_size, head_size, dropout):
        super().__init__()
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        # Máscara triangular: cada token solo ve el pasado, no el futuro
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Varias cabezas de atención en paralelo — cada una aprende distintas relaciones."""
    def __init__(self, d_model, block_size, n_head, dropout):
        super().__init__()
        head_size = d_model // n_head
        self.heads = nn.ModuleList([
            HeadAtencion(d_model, block_size, head_size, dropout) for _ in range(n_head)
        ])
        self.proj    = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Red feed-forward posición a posición (procesa cada token por separado)."""
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BloqueTransformer(nn.Module):
    """Un bloque Transformer = MultiHeadAttention + FeedForward + LayerNorm residual."""
    def __init__(self, d_model, block_size, n_head, dropout):
        super().__init__()
        self.sa   = MultiHeadAttention(d_model, block_size, n_head, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # atención con conexión residual
        x = x + self.ffwd(self.ln2(x))  # feed-forward con conexión residual
        return x


class ModeloLLM(nn.Module):
    """
    Modelo de Lenguaje Causal estilo GPT.
    Embedding de tokens + Embedding posicional → N bloques Transformer → cabeza de lenguaje.
    """
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.token_embedding_table    = nn.Embedding(vocab_size, config.D_MODEL)
        self.position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.D_MODEL)
        self.blocks = nn.Sequential(*[
            BloqueTransformer(config.D_MODEL, config.BLOCK_SIZE, config.N_HEAD, config.DROPOUT)
            for _ in range(config.N_LAYER)
        ])
        self.ln_f   = nn.LayerNorm(config.D_MODEL)
        self.lm_head = nn.Linear(config.D_MODEL, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T), ignore_index=-100)

        return logits, loss

    def generar(self, idx, max_new_tokens, temperature=0.7, top_k=10, eos_id=None):
        """
        Genera tokens autoregresivamente y se detiene al encontrar <EOS>.

        temperature: < 1.0 hace la distribución más "afilada" (respuestas más deterministas).
        top_k: solo muestrea entre los k tokens más probables (reduce incoherencias).
        eos_id: token especial de fin de secuencia; detiene la generación al aparecer.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and idx_next.item() == eos_id:
                break

        return idx
