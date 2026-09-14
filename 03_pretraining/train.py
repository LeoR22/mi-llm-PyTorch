"""
train.py — Bucle de pre-entrenamiento causal (predicción del siguiente token).

El modelo aprende la estructura del idioma prediciendo token a token,
sin necesitar etiquetas humanas — solo texto crudo.
"""

import math
import torch
from config import ConfigLLM
from model import ModeloLLM


def ejecutar_preentrenamiento(modelo, tokenizer, texto_corpus):
    tokens      = tokenizer.encode(texto_corpus, add_special_tokens=False)
    data_tensor = torch.tensor(tokens, dtype=torch.long)

    if len(data_tensor) <= ConfigLLM.BLOCK_SIZE:
        raise ValueError(
            f"Corpus demasiado corto tras tokenizar: {len(data_tensor)} tokens. "
            f"Se necesitan al menos {ConfigLLM.BLOCK_SIZE + 1}. "
            "Verifica que Wikipedia descargó datos o amplía el corpus de respaldo."
        )

    def get_batch():
        ix = torch.randint(len(data_tensor) - ConfigLLM.BLOCK_SIZE, (ConfigLLM.BATCH_SIZE,))
        x  = torch.stack([data_tensor[i:i + ConfigLLM.BLOCK_SIZE] for i in ix])
        y  = torch.stack([data_tensor[i + 1:i + ConfigLLM.BLOCK_SIZE + 1] for i in ix])
        return x.to(ConfigLLM.DEVICE), y.to(ConfigLLM.DEVICE)

    optimizer = torch.optim.AdamW(modelo.parameters(), lr=ConfigLLM.LEARNING_RATE)

    modelo.train()
    print("\n[03_pretraining] Iniciando Pre-Entrenamiento Causal...")
    for step in range(ConfigLLM.EPOCHS_PRETRAIN):
        xb, yb     = get_batch()
        logits, loss = modelo(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0 or step == 0:
            perplejidad = math.exp(loss.item())
            print(f"  Paso {step+1}/{ConfigLLM.EPOCHS_PRETRAIN} | "
                  f"Loss: {loss.item():.4f} | Perplejidad: {perplejidad:.2f}")

    return modelo
