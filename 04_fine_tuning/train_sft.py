"""
train_sft.py — Fine-Tuning Supervisado (SFT).

Ajusta el modelo pre-entrenado con pares pregunta/respuesta para que adopte
el formato de asistente conversacional en español colombiano.
"""

import torch
from sft_dataset import obtener_sft_tensors


def ejecutar_sft(modelo, tokenizer, config):
    x_sft, y_sft = obtener_sft_tensors(tokenizer, config)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=config.LEARNING_RATE)

    modelo.train()
    print("\n[04_fine_tuning] Iniciando Fine-Tuning Supervisado (SFT)...")
    for step in range(config.EPOCHS_SFT):
        logits, loss = modelo(x_sft, y_sft)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0 or step == 0:
            print(f"  Paso SFT {step+1}/{config.EPOCHS_SFT} | Loss SFT: {loss.item():.4f}")

    return modelo
