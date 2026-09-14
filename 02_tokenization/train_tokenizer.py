import json

class TokenizadorBPESimple:
    """Implementación BPE educativa a nivel de subpalabras."""
    def __init__(self, target_vocab_size=150):
        self.target_vocab_size = target_vocab_size
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.merges = {}
        self.vocab = {}

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text):
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)
        vocab_size = 256
        num_merges = self.target_vocab_size - vocab_size - len(self.special_tokens)
        
        idx = 256
        for _ in range(max(0, num_merges)):
            stats = self._get_stats(ids)
            if not stats:
                break
            best = max(stats, key=stats.get)
            ids = self._merge(ids, best, idx)
            self.merges[best] = idx
            idx += 1

        self.vocab = {i: bytes([i]) for i in range(256)}
        for p0, p1 in self.merges:
            self.vocab[self.merges[(p0, p1)]] = self.vocab[p0] + self.vocab[p1]
            
        self.pad_id = idx
        self.unk_id = idx + 1
        self.bos_id = idx + 2
        self.eos_id = idx + 3

    def encode(self, text, add_special_tokens=True):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
            
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        byte_list = []
        for idx in ids:
            if idx in self.vocab:
                byte_list.append(self.vocab[idx])
        return bytes(b"".join(byte_list)).decode("utf-8", errors="replace")

def entrenar_y_guardar_tokenizador(ruta_corpus="data_clean.txt"):
    with open(ruta_corpus, "r", encoding="utf-8") as f:
        texto = f.read()
        
    tok = TokenizadorBPESimple(target_vocab_size=200)
    tok.train(texto)
    print(f"[02_tokenization] Tokenizador BPE entrenado. Tamaño de vocabulario: {tok.eos_id + 1}")
    return tok

if __name__ == "__main__":
    entrenar_y_guardar_tokenizador()