import torch

class ConfigLLM:
    """Configuración centralizada de hiperparámetros para el modelo y entrenamiento."""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 4
    BLOCK_SIZE = 64      # Longitud máxima de contexto en tokens
    D_MODEL = 128        # Dimensión de los embeddings (representaciones ocultas)
    N_HEAD = 4           # Número de cabezas de atención multi-head
    N_LAYER = 4          # Capas Transformer apiladas
    DROPOUT = 0.1
    LEARNING_RATE = 1e-3
    EPOCHS_PRETRAIN = 1000
    EPOCHS_SFT = 800
