"""
sft_dataset.py — Datos instruccionales en español colombiano para el Fine-Tuning Supervisado.

Cada par pregunta/respuesta le enseña al modelo el formato:
  "usuario: <pregunta> asistente: <respuesta>"

El loss se enmascara con -100 en la parte del prompt para que el gradiente
solo fluya por las palabras de la respuesta del asistente.
"""

import torch

DATOS_INSTRUCCIONALES = [
    {"p": "que es el procesamiento del lenguaje natural",
     "r": "es una rama de la inteligencia artificial que le permite a los computadores entender y generar texto en lenguaje humano."},
    {"p": "para que sirven los transformers",
     "r": "los transformers usan capas de atencion para procesar secuencias de texto y son la base de modelos como chatgpt."},
    {"p": "que es pytorch",
     "r": "pytorch es una libreria de python para desarrollar y entrenar redes neuronales, muy usada en investigacion de inteligencia artificial."},
    {"p": "que es un tokenizador",
     "r": "un tokenizador convierte el texto en numeros llamados tokens para que el modelo lo pueda procesar."},
    {"p": "que es el preentrenamiento",
     "r": "el preentrenamiento es cuando el modelo aprende la estructura del idioma leyendo mucho texto sin etiquetar."},
    {"p": "que es la inteligencia artificial",
     "r": "la inteligencia artificial es la rama de la informatica que crea sistemas capaces de aprender, razonar y tomar decisiones."},
    {"p": "que es un transformer",
     "r": "un transformer es una red neuronal con capas de autoatencion y redes feed-forward, es la arquitectura detras de los modelos de lenguaje modernos."},
    {"p": "que es el fine tuning",
     "r": "el fine tuning es ajustar un modelo ya entrenado con ejemplos especificos de preguntas y respuestas para que se comporte como un asistente."},
    {"p": "que es una red neuronal",
     "r": "una red neuronal es un sistema inspirado en el cerebro humano, formado por capas de neuronas artificiales que aprenden patrones a partir de datos."},
    {"p": "que es el aprendizaje profundo",
     "r": "el aprendizaje profundo o deep learning usa redes neuronales con muchas capas para aprender representaciones complejas de los datos."},
    {"p": "que es python",
     "r": "python es el lenguaje de programacion mas usado para inteligencia artificial por su simplicidad y la gran cantidad de librerias disponibles."},
    {"p": "como funciona la atencion en transformers",
     "r": "la atencion permite que cada palabra del texto mire a las demas y decida cuales son mas importantes para entender el contexto."},
]


def obtener_sft_tensors(tokenizer, config):
    inputs, targets = [], []
    for item in DATOS_INSTRUCCIONALES:
        prompt_enc = tokenizer.encode(
            f"usuario: {item['p']} asistente: ", add_special_tokens=False
        )
        resp_enc = tokenizer.encode(item['r'], add_special_tokens=False) + [tokenizer.eos_id]

        seq = prompt_enc + resp_enc
        if len(seq) > config.BLOCK_SIZE:
            seq = seq[:config.BLOCK_SIZE]

        x = seq[:-1]
        y = seq[1:]

        # Enmascarar el prompt con -100: el gradiente solo fluye por la respuesta
        mask_len = min(len(prompt_enc) - 1, len(x))
        y_masked = [-100] * mask_len + y[mask_len:]

        pad_len  = config.BLOCK_SIZE - len(x)
        x        = x + [tokenizer.pad_id] * pad_len
        y_masked = y_masked + [-100] * pad_len

        inputs.append(torch.tensor(x, dtype=torch.long))
        targets.append(torch.tensor(y_masked, dtype=torch.long))

    return (
        torch.stack(inputs).to(config.DEVICE),
        torch.stack(targets).to(config.DEVICE),
    )
