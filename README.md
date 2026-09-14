# Mi LLM de Asistente de Inteligencia Artificial en Español desde Cero 🚀

Guía práctica para construir, entrenar y desplegar un Modelo Grande de Lenguaje (LLM) en español **desde cero** usando PyTorch. Basado en [How to Build Your Own Language-Specific LLM](https://www.freecodecamp.org/news/how-to-build-your-own-language-specific-llm-handbook/).

> **¿Para quién es esto?** Para cualquier persona curiosa sobre cómo funcionan ChatGPT, Gemini o LLaMA por dentro. No necesitas ser experto — si sabes Python básico, puedes seguir este proyecto paso a paso.

---

## ¿Qué vas a aprender?

| Etapa | Concepto clave | ¿Para qué sirve? |
|-------|---------------|------------------|
| Preparación de datos | Normalización Unicode, expresiones regulares | Limpiar texto crudo antes de entrenar |
| Tokenización | Algoritmo Byte-Pair Encoding (BPE) | Convertir texto en números que el modelo entiende |
| Modelo | Arquitectura Transformer Causal (estilo GPT) | La red neuronal que aprende el idioma |
| Pre-entrenamiento | Modelado de lenguaje causal | El modelo lee mucho texto y aprende a predecir la siguiente palabra |
| Fine-Tuning (SFT) | Alineación supervisada | El modelo aprende a responder preguntas como un asistente |
| Despliegue | Interfaz web con Gradio | Chatear con tu modelo en el navegador |

---

## 📁 Estructura del Proyecto

```
llm-idioma/
├── 01_data_preparation/       ← PASO 1: conseguir y limpiar texto
│   ├── download_data.py       # Descarga artículos de Wikipedia en español
│   └── clean_data.py          # Quita caracteres raros, normaliza el texto
│
├── 02_tokenization/           ← PASO 2: convertir texto en números
│   ├── train_tokenizer.py     # Implementación del algoritmo BPE desde cero
│   └── test_tokenizer.py      # Prueba que el tokenizador encode/decode bien
│
├── 03_pretraining/            ← PASO 3: enseñarle el idioma al modelo
│   ├── config.py              # Todos los parámetros del modelo en un solo lugar
│   ├── model.py               # La arquitectura Transformer (el cerebro del LLM)
│   └── train.py               # El bucle que entrena el modelo con texto crudo
│
├── 04_fine_tuning/            ← PASO 4: enseñarle a responder preguntas
│   ├── sft_dataset.py         # Pares pregunta/respuesta para el entrenamiento
│   └── train_sft.py           # Ajusta el modelo para que actúe como asistente
│
├── 05_deployment/             ← PASO 5: publicar el modelo en una UI web
│   └── app.py                 # Pipeline completo + interfaz con Gradio
│
├── requirements.txt           ← Librerías necesarias (torch, gradio, etc.)
└── README.md                  ← Este archivo
```

---

## 🚀 Cómo Ejecutar

### 1. Crear entorno virtual e instalar dependencias

Un entorno virtual aísla las librerías del proyecto para que no entren en conflicto con otros proyectos de Python en tu máquina.

```bash
python -m venv .venv

# Activar el entorno — Windows:
.venv\Scripts\activate

# Activar el entorno — macOS/Linux:
source .venv/bin/activate

# Instalar librerías
pip install -r requirements.txt
```

### 2. (Opcional) Explorar cada módulo por separado

Puedes correr cada paso individualmente para entender qué hace antes de ejecutar todo junto.

```bash
# Descargar corpus de Wikipedia en español
python 01_data_preparation/download_data.py

# Limpiar y normalizar el texto
python 01_data_preparation/clean_data.py

# Probar el tokenizador BPE
python 02_tokenization/test_tokenizer.py
```

### 3. Lanzar la aplicación web completa

Este comando ejecuta **todo el pipeline automáticamente**: descarga datos, entrena el tokenizador, pre-entrena el modelo, aplica fine-tuning y lanza la UI.

```bash
python 05_deployment/app.py
```

El entrenamiento tarda entre 1 y 3 minutos en CPU. Cuando veas esto en la terminal:

```
Running on local URL:  http://127.0.0.1:7860
```

Copia ese enlace, ábrelo en tu navegador y chatea con tu modelo.

---

## 🧠 Conceptos Explicados

### ¿Qué es un LLM?

Un **Modelo Grande de Lenguaje** (LLM) es una red neuronal entrenada para predecir qué palabra viene después de un texto dado. Si lo entrenas con suficientes datos, aprende gramática, hechos, razonamiento y hasta cómo responder preguntas — todo solo prediciendo la siguiente palabra.

---

### Paso 1 — Preparación de datos

Antes de entrenar cualquier modelo, el texto crudo necesita limpieza:

- **Normalización Unicode (NFKC):** En Unicode, la letra `é` puede representarse de dos formas distintas (como un solo carácter o como `e` + acento combinado). NFKC unifica estas variantes para que el tokenizador las trate igual.
- **Expresiones regulares:** Se eliminan URLs, etiquetas HTML, caracteres especiales y espacios dobles.
- **Lowercase:** Convertir todo a minúsculas reduce el vocabulario necesario (`"Python"` y `"python"` se vuelven el mismo token).

---

### Paso 2 — Tokenización (BPE)

El modelo no entiende texto, solo números. El tokenizador convierte texto → lista de enteros y viceversa.

**¿Por qué no simplemente un número por letra?**
Porque el vocabulario sería enorme para idiomas con muchas variantes. BPE es más inteligente:

```
Inicio:  cada byte es un token  →  256 tokens posibles
Paso 1:  el par de bytes más frecuente se fusiona  →  257 tokens
Paso 2:  el siguiente par más frecuente se fusiona  →  258 tokens
...
Continúa hasta alcanzar el vocab_size deseado
```

Resultado: palabras comunes como `"que"` quedan como 1 solo token. Palabras raras se dividen en partes conocidas: `"tokenización"` → `["token", "iza", "ción"]`.

**Tokens especiales:**
- `<PAD>` — relleno para igualar longitudes en un batch
- `<BOS>` — inicio de secuencia (Begin Of Sequence)
- `<EOS>` — fin de secuencia (End Of Sequence)
- `<UNK>` — token desconocido

---

### Paso 3 — Arquitectura Transformer

El Transformer es la arquitectura detrás de GPT, LLaMA, Gemini y casi todos los LLMs modernos. Se compone de bloques apilados:

```
Texto tokenizado
      ↓
[Embedding de tokens]      → convierte cada número en un vector de D_MODEL dimensiones
      +
[Embedding posicional]     → le dice al modelo en qué posición está cada token
      ↓
┌─────────────────────────────────────────┐
│  Bloque Transformer × N_LAYER veces:    │
│                                         │
│  ┌─ LayerNorm                           │
│  ├─ Multi-Head Attention  ←── CLAVE     │
│  ├─ + conexión residual                 │
│  ├─ LayerNorm                           │
│  ├─ FeedForward                         │
│  └─ + conexión residual                 │
└─────────────────────────────────────────┘
      ↓
[LayerNorm final]
      ↓
[Capa lineal → vocab_size]  → probabilidad de cada token como siguiente
```

**¿Qué es la Atención?**
Es el mecanismo que le permite al modelo, al procesar la palabra `"banco"`, mirar todas las demás palabras del contexto y decidir cuáles son relevantes para entender si es un banco de dinero o un banco para sentarse. Cada cabeza de atención aprende a relacionar palabras de formas distintas.

**¿Por qué Multi-Head?**
Varias cabezas en paralelo capturan distintos tipos de relaciones al mismo tiempo: sintácticas, semánticas, de referencia, etc.

**¿Qué es la máscara triangular (`tril`)?**
Impide que el modelo "haga trampa" viendo tokens futuros al entrenar. Si está prediciendo el token en posición 5, solo puede ver las posiciones 1 a 4.

**¿Qué es la conexión residual?**
En vez de `x = capa(x)`, se hace `x = x + capa(x)`. Esto permite que el gradiente fluya sin desvanecerse en redes profundas, haciendo posible apilar muchas capas.

---

### Paso 4 — Pre-entrenamiento

El modelo aprende el idioma leyendo texto crudo sin necesitar etiquetas humanas:

```
Entrada:  ["el", "modelo", "aprende", "el"]
Target:   ["modelo", "aprende", "el", "idioma"]
```

Por cada token, el modelo predice el siguiente. La **pérdida (loss)** mide qué tan equivocado estuvo. El optimizador **AdamW** ajusta los millones de parámetros del modelo para minimizar esa pérdida. La **perplejidad** es `e^loss` — cuanto más baja, más "seguro" está el modelo de sus predicciones.

---

### Paso 5 — Fine-Tuning Supervisado (SFT)

El modelo ya sabe español del pre-entrenamiento. Ahora aprende el **formato de asistente**:

```
Secuencia completa:
"usuario: qué es pytorch asistente: es una librería de python para redes neuronales"

Loss mask:
 -100  -100  -100  -100  -100  -100    1     1    1    1    1    1    1    1
 ↑ se ignora el prompt ↑              ↑ solo aprende a generar la respuesta ↑
```

El `-100` le dice a PyTorch que ignore esas posiciones al calcular el gradiente. Así el modelo no "pierde tiempo" aprendiendo a repetir la pregunta — solo aprende a generar buenas respuestas.

---

### Paso 6 — Despliegue con Gradio

`app.py` ejecuta todo el pipeline y expone una interfaz web. Usa una estrategia en dos pasos para responder:

1. **Recuperación por similitud** — busca la pregunta más parecida en el dataset SFT usando similitud de bigramas (Jaccard). Si la coincidencia es alta, devuelve la respuesta memorizada. Esto garantiza respuestas correctas para las preguntas entrenadas.

2. **Generación con el modelo** — para preguntas nuevas, el Transformer genera tokens uno a uno hasta encontrar `<EOS>` o alcanzar el límite de tokens.

---

## ⚙️ Hiperparámetros (config.py)

| Parámetro | Valor actual | ¿Qué controla? | Para un modelo más grande |
|-----------|-------------|----------------|--------------------------|
| `BLOCK_SIZE` | 64 | Cuántos tokens puede ver el modelo a la vez (contexto) | GPT-4: 128,000 |
| `D_MODEL` | 128 | Tamaño del vector que representa cada token | LLaMA 7B: 4,096 |
| `N_HEAD` | 4 | Cabezas de atención en paralelo | LLaMA 7B: 32 |
| `N_LAYER` | 4 | Capas Transformer apiladas | LLaMA 7B: 32 |
| `DROPOUT` | 0.1 | Regularización — apaga neuronas al azar durante entrenamiento | Igual |
| `LEARNING_RATE` | 1e-3 | Qué tan grandes son los pasos del optimizador | 3e-4 con scheduler |
| `EPOCHS_PRETRAIN` | 1000 | Pasos de pre-entrenamiento | Billones de tokens |
| `EPOCHS_SFT` | 800 | Pasos de fine-tuning | Miles de ejemplos |

---

## ⚠️ Limitaciones (intencionales para aprendizaje)

| Limitación | Por qué existe | Cómo superarla |
|---|---|---|
| Corpus pequeño (~30K chars) | Para entrenar rápido en CPU | Usar Common Crawl, Wikipedia completa |
| Modelo diminuto (128 dims) | Cabe en cualquier máquina | Subir `D_MODEL` y `N_LAYER`, usar GPU |
| No persiste en disco | Simplifica el código | Agregar `torch.save` / `torch.load` |
| Vocabulario BPE propio | Para entender cómo funciona | Usar SentencePiece o tiktoken en producción |
| Sin GPU obligatoria | Accesible para todos | Agregar `.to("cuda")` con una GPU NVIDIA |

---

## 🔭 ¿Cómo escalar esto a un LLM real?

Si quieres ir más allá, este es el camino:

1. **nanoGPT** (Andrej Karpathy) — este mismo proyecto con optimizaciones de producción
2. **LLaMA** — agrega RoPE, RMSNorm y SwiGLU sobre la base de nanoGPT
3. **Hugging Face Transformers** — para fine-tuning con datasets reales sin partir de cero

Las mejoras más importantes para un modelo más grande:

```
Datos:         Common Crawl + Wikipedia + libros (decenas de GB)
Arquitectura:  RoPE, RMSNorm, SwiGLU, Flash Attention, KV-cache
Entrenamiento: bf16, gradient checkpointing, múltiples GPUs (DDP/FSDP)
Alineación:    RLHF o DPO después del SFT
```
## Licencia

Este proyecto está licenciado bajo la Licencia MIT.

## Contacto

- Leandro Rivera: leo.232rivera@gmail.com
- LinkedIn: https://www.linkedin.com/in/leandrorivera/

### ¡Feliz Codificación! 🚀

Si encuentras útil este proyecto, ¡dale una ⭐ en GitHub! 😊
