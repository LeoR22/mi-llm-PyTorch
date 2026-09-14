from train_tokenizer import entrenar_y_guardar_tokenizador

if __name__ == "__main__":
    tok = entrenar_y_guardar_tokenizador()
    frase = "inteligencia artificial"
    tokens = tok.encode(frase)
    print(f"Frase original: {frase}")
    print(f"Tokens codificados: {tokens}")
    print(f"Texto decodificado: {tok.decode(tokens)}")