# Mi LLM de Asistente de Inteligencia Artificial en Español desde Cero 🚀

Guía práctica para construir, entrenar y desplegar un Modelo Grande de Lenguaje (LLM) en español **desde cero** usando PyTorch. Basado en [How to Build Your Own Language-Specific LLM](https://www.freecodecamp.org/news/how-to-build-your-own-language-specific-llm-handbook/).

---

## ¿Qué vas a aprender?

| Etapa | Concepto clave |
|-------|---------------|
| Preparación de datos | Normalización Unicode, expresiones regulares, corpus de texto |
| Tokenización | Algoritmo Byte-Pair Encoding (BPE) implementado a mano |
| Modelo | Arquitectura Transformer Causal (estilo GPT) con PyTorch |
| Pre-entrenamiento | Modelado de lenguaje causal — predecir el siguiente token |
| Fine-Tuning (SFT) | Alineación supervisada para formato pregunta-respuesta |
| Despliegue | Interfaz web con Gradio |

---

## 📁 Estructura del Proyecto

```
llm-idioma/
├── 01_data_preparation/
│   ├── download_data.py   # Genera el corpus en español (texto crudo)
│   └── clean_data.py      # Limpia y normaliza el texto
│
├── 02_tokenization/
│   ├── train_tokenizer.py # Implementación BPE educativa
│   └── test_tokenizer.py  # Prueba encode/decode
│
├── 03_pretraining/
│   ├── config.py          # Hiperparámetros centralizados
│   ├── model.py           # Arquitectura Transformer (HeadAtencion → ModeloLLM)
│   └── train.py           # Bucle de pre-entrenamiento causal
│
├── 04_fine_tuning/
│   ├── sft_dataset.py     # Datos instruccionales con enmascaramiento de loss
│   ├── train_sft.py       # Bucle de fine-tuning supervisado
│   └── train.py           # (re-exporta ejecutar_preentrenamiento)
│
├── 05_deployment/
│   └── app.py             # Pipeline completo + UI con Gradio
│
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo Ejecutar

### 1. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. (Opcional) Explorar cada módulo por separado

```bash
# Generar corpus crudo
python 01_data_preparation/download_data.py

# Limpiar texto
python 01_data_preparation/clean_data.py

# Probar el tokenizador BPE
python 02_tokenization/test_tokenizer.py
```

### 3. Lanzar la aplicación web completa

```bash
python 05_deployment/app.py
```

La app ejecuta **todo el pipeline automáticamente** (datos → tokenizador → preentrenamiento → SFT) y luego abre el servidor. Cuando veas en la terminal:

```
Running on local URL:  http://127.0.0.1:7860
```

Abre ese enlace en tu navegador y chatea con tu modelo.

---

## 🧠 Conceptos Explicados

### Byte-Pair Encoding (BPE)
El texto se convierte primero en bytes (256 valores posibles). Luego, iterativamente se fusionan los pares de bytes más frecuentes hasta alcanzar el tamaño de vocabulario deseado. Esto permite representar palabras raras como combinación de subpalabras conocidas.

### Transformer Causal
El modelo aprende a predecir el siguiente token dado el contexto anterior. La máscara triangular (`tril`) impide que cada posición "vea" tokens futuros — esto es lo que hace que sea *causal*.

### Fine-Tuning Supervisado (SFT)
Se entrena el modelo pre-entrenado con pares pregunta/respuesta. El truco clave es enmascarar con `-100` la parte del prompt en el target, para que el gradiente solo fluya por la respuesta del asistente.

---

## ⚙️ Hiperparámetros (config.py)

| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `BLOCK_SIZE` | 64 | Longitud máxima de contexto en tokens |
| `D_MODEL` | 128 | Dimensión de los embeddings |
| `N_HEAD` | 4 | Cabezas de atención multi-head |
| `N_LAYER` | 4 | Capas Transformer apiladas |
| `EPOCHS_PRETRAIN` | 300 | Pasos de pre-entrenamiento |
| `EPOCHS_SFT` | 200 | Pasos de fine-tuning |

---

## ⚠️ Limitaciones (intencionales para aprendizaje)

- El corpus es muy pequeño (10 oraciones) — las respuestas serán imperfectas.
- No se usa GPU obligatoriamente — funciona en CPU para que cualquier máquina pueda ejecutarlo.
- El modelo no persiste en disco — se re-entrena cada vez que arrancas `app.py`.

Para extenderlo: agrega más frases en `CORPUS_ESPANOL` (download_data.py) y más pares en `DATOS_INSTRUCCIONALES` (sft_dataset.py).
