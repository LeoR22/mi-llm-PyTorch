import os
import re
import unicodedata

def limpiar_texto_espanol(texto: str) -> str:
    """Aplica normalización Unicode NFKC y filtrado por expresiones regulares."""
    texto = unicodedata.normalize('NFKC', texto)
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r'<[^>]+>', '', texto)
    texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9\s\.,\?!]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip().lower()

def procesar_y_limpiar(ruta_entrada="data_raw.txt", ruta_salida="data_clean.txt"):
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No se encontró el archivo {ruta_entrada}")

    with open(ruta_entrada, "r", encoding="utf-8") as f_in:
        lineas = f_in.readlines()

    lineas_limpias = [limpiar_texto_espanol(linea) for linea in lineas if linea.strip()]

    with open(ruta_salida, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lineas_limpias))
        
    print(f"[01_data_preparation] Texto limpio guardado en: {ruta_salida}")

if __name__ == "__main__":
    procesar_y_limpiar()