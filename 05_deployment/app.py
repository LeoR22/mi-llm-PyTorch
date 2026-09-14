"""
app.py — Interfaz web interactiva con Gradio para el LLM en Español Colombiano.

Pipeline completo que se ejecuta al arrancar:
  1. Descarga artículos de Wikipedia en español (o usa corpus de respaldo)
  2. Limpia y normaliza el texto
  3. Entrena el tokenizador BPE
  4. Pre-entrena el modelo Transformer (lenguaje causal)
  5. Aplica Fine-Tuning Supervisado (SFT) con respuestas en español colombiano
  6. Lanza la interfaz web en http://127.0.0.1:7860
"""

import gradio as gr
import torch
import sys
import os

# ---------------------------------------------------------------------------
# Rutas absolutas — el script funciona desde cualquier directorio
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW   = os.path.join(BASE_DIR, "data_raw.txt")
DATA_CLEAN = os.path.join(BASE_DIR, "data_clean.txt")

for sub in ["01_data_preparation", "02_tokenization", "03_pretraining", "04_fine_tuning"]:
    sys.path.insert(0, os.path.join(BASE_DIR, sub))

from download_data import descargar_corpus_crudo
from clean_data import limpiar_texto_espanol, procesar_y_limpiar
from train_tokenizer import TokenizadorBPESimple
from config import ConfigLLM
from model import ModeloLLM
from train import ejecutar_preentrenamiento
from train_sft import ejecutar_sft
from sft_dataset import DATOS_INSTRUCCIONALES

# ---------------------------------------------------------------------------
# 1 & 2 — Datos
# ---------------------------------------------------------------------------
print("[05_deployment] Descargando y preparando datos...")
descargar_corpus_crudo(ruta_salida=DATA_RAW)
procesar_y_limpiar(ruta_entrada=DATA_RAW, ruta_salida=DATA_CLEAN)

with open(DATA_CLEAN, "r", encoding="utf-8") as f:
    texto_corpus = f.read()

# ---------------------------------------------------------------------------
# 3 — Tokenizador BPE (tamaño proporcional al corpus)
# ---------------------------------------------------------------------------
vocab_objetivo = min(1000, max(300, len(texto_corpus) // 20))
print(f"[05_deployment] Entrenando tokenizador BPE (vocab objetivo: {vocab_objetivo})...")
tokenizer = TokenizadorBPESimple(target_vocab_size=vocab_objetivo)
tokenizer.train(texto_corpus)
vocab_size = tokenizer.eos_id + 1
print(f"[05_deployment] Vocabulario final: {vocab_size} tokens")

# ---------------------------------------------------------------------------
# 4 — Pre-entrenamiento
# ---------------------------------------------------------------------------
modelo = ModeloLLM(vocab_size, ConfigLLM).to(ConfigLLM.DEVICE)
modelo = ejecutar_preentrenamiento(modelo, tokenizer, texto_corpus)

# ---------------------------------------------------------------------------
# 5 — Fine-Tuning Supervisado
# ---------------------------------------------------------------------------
modelo = ejecutar_sft(modelo, tokenizer, ConfigLLM)

print("\n[05_deployment] Modelo listo. Iniciando servidor Gradio...")

# ---------------------------------------------------------------------------
# Helpers de inferencia
# ---------------------------------------------------------------------------

def _similitud(a: str, b: str) -> float:
    """
    Similitud por bigramas de caracteres (Jaccard).
    Permite encontrar la pregunta más parecida en el dataset SFT sin necesitar
    embeddings ni librerías externas — ideal para aprendizaje.
    """
    def bigramas(s):
        s = s.lower().strip()
        return set(s[i:i+2] for i in range(len(s) - 1))
    bg_a, bg_b = bigramas(a), bigramas(b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


def _es_texto_valido(texto: str) -> bool:
    """Verifica que el texto tenga al menos 60% de caracteres latinos válidos."""
    if not texto or len(texto) < 5:
        return False
    validos = sum(1 for c in texto if c.isalpha() or c in " .,;:¿?¡!")
    return (validos / len(texto)) >= 0.6


# ---------------------------------------------------------------------------
# Función de chat
# ---------------------------------------------------------------------------

def responder_chat(mensaje: str, historial: list) -> str:
    """
    Estrategia en dos pasos (RAG simplificado):

    1. Recuperación: busca la pregunta más similar en los datos SFT.
       Si la similitud ≥ 0.25, devuelve la respuesta memorizada (siempre correcta).

    2. Generación: si no hay coincidencia, usa el modelo con decodificación
       greedy (top_k=1). Si la salida es incoherente, avisa al usuario.
    """
    p_limpia = limpiar_texto_espanol(mensaje)

    # Paso 1 — recuperación por similitud
    mejor_score, mejor_respuesta = 0.0, None
    for item in DATOS_INSTRUCCIONALES:
        score = _similitud(p_limpia, item["p"])
        if score > mejor_score:
            mejor_score, mejor_respuesta = score, item["r"]

    if mejor_score >= 0.25 and mejor_respuesta:
        return mejor_respuesta

    # Paso 2 — generación con el modelo
    modelo.eval()
    prompt = f"usuario: {p_limpia} asistente: "
    tokens_input = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long,
    ).to(ConfigLLM.DEVICE)

    with torch.no_grad():
        out_tokens = modelo.generar(
            tokens_input,
            max_new_tokens=60,
            temperature=0.2,
            top_k=1,
            eos_id=tokenizer.eos_id,
        )

    texto_generado = tokenizer.decode(out_tokens[0].tolist())
    if "asistente:" in texto_generado:
        respuesta = texto_generado.split("asistente:")[-1].strip()
    else:
        respuesta = texto_generado.strip()

    if _es_texto_valido(respuesta):
        return respuesta

    return (
        "Ese tema no está en mi corpus de entrenamiento todavía. "
        "Pregúntame sobre: inteligencia artificial, transformers, pytorch, "
        "tokenización, redes neuronales o procesamiento del lenguaje natural, ¡y te cuento!"
    )


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=responder_chat,
    title="Asistente de Inteligencia Artificial en Español",
    description=(
        "Modelo de Lenguaje construido desde cero con PyTorch — arquitectura Transformer, "
        "tokenizador BPE y fine-tuning supervisado.\n\n"
        "Pregunta sobre inteligencia artificial, redes neuronales, transformers, "
        "tokenización o aprendizaje profundo."
    ),
    examples=[
        "¿qué es pytorch?",
        "¿qué es un transformer?",
        "¿qué es una red neuronal?",
        "¿qué es la inteligencia artificial?",
        "¿cómo funciona la atención en transformers?",
    ],
)

if __name__ == "__main__":
    demo.launch()
