"""
download_data.py — Descarga artículos de Wikipedia en español sobre IA/NLP/ML.

Los artículos se concatenan en data_raw.txt. Si Wikipedia no está disponible
(sin internet) se usa el corpus sintético de respaldo.
"""

import os
import time

# Temas a descargar — enfocados en el dominio del proyecto
TEMAS_WIKIPEDIA = [
    "Inteligencia artificial",
    "Procesamiento de lenguajes naturales",
    "Aprendizaje automático",
    "Red neuronal artificial",
    "Aprendizaje profundo",
    "Transformador (modelo de lenguaje)",
    "PyTorch",
    "Tokenización",
    "Representación distribuida de palabras",
    "Modelo de lenguaje",
]

# Corpus de respaldo por si no hay conexión a internet
CORPUS_RESPALDO = [
    "El procesamiento del lenguaje natural es un campo de la inteligencia artificial.",
    "Los modelos de lenguaje grandes utilizan redes neuronales profundas para generar texto.",
    "El mecanismo de atencion permite a la red relacionar diferentes palabras en un contexto.",
    "Un tokenizador convierte texto continuo en una secuencia de numeros llamados tokens.",
    "El entrenamiento supervisado ajusta los pesos del modelo usando ejemplos de preguntas y respuestas.",
    "La inteligencia artificial esta transformando la ciencia, la tecnologia y la sociedad.",
    "Python es el lenguaje de programacion mas utilizado para desarrollar redes neuronales.",
    "PyTorch facilita la creacion de grafos de computacion dinamicos y el calculo de gradientes.",
    "Un Transformer se compone de capas de autoatencion y redes feed-forward.",
    "El preentrenamiento permite que el modelo aprenda la estructura sintactica y semantica del idioma.",
]


def descargar_corpus_crudo(ruta_salida="data_raw.txt"):
    """Descarga artículos de Wikipedia en español y los guarda en disco."""
    try:
        import wikipediaapi
    except ImportError:
        print("[01_data_preparation] ⚠ wikipedia-api no instalado. Usando corpus de respaldo.")
        _guardar_respaldo(ruta_salida)
        return

    wiki = wikipediaapi.Wikipedia(
        language="es",
        user_agent="llm-idioma-educativo/1.0"
    )

    textos = []
    for tema in TEMAS_WIKIPEDIA:
        print(f"[01_data_preparation] Descargando: {tema}...", end=" ", flush=True)
        pagina = wiki.page(tema)
        if pagina.exists():
            # Tomar los primeros 3000 caracteres de cada artículo
            fragmento = pagina.text[:3000].strip()
            textos.append(fragmento)
            print(f"✓ ({len(fragmento)} caracteres)")
        else:
            print("✗ no encontrado")
        time.sleep(0.3)  # pausa para no saturar la API

    if not textos:
        print("[01_data_preparation] ⚠ No se descargó ningún artículo. Usando corpus de respaldo.")
        _guardar_respaldo(ruta_salida)
        return

    corpus_completo = "\n\n".join(textos)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(corpus_completo)

    total_chars = len(corpus_completo)
    print(f"\n[01_data_preparation] ✓ Corpus guardado en '{ruta_salida}' "
          f"({len(textos)} artículos, {total_chars:,} caracteres)")


def _guardar_respaldo(ruta_salida):
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for linea in CORPUS_RESPALDO:
            f.write(linea + "\n")
    print(f"[01_data_preparation] Corpus de respaldo guardado en: {ruta_salida}")


if __name__ == "__main__":
    descargar_corpus_crudo()
