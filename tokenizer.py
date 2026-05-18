from __future__ import annotations

class Tokenizer:
    """
    Character-level tokenizer for:
      digits: '0'..'9'
      specials: 'N','B','D','O','E'
    """
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()

    def _add(self, ch: str):
        idx = len(self.vocab)
        self.vocab[ch] = idx
        self.reverse_vocab[idx] = ch

    def _build_vocab(self):
        for i in range(10):
            self._add(str(i))
        for ch in ["N", "B", "D", "O", "E"]:
            self._add(ch)

    def encode(self, text: str) -> list[int]:
        return [self.vocab[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.reverse_vocab[t] for t in tokens)

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def eos_token_id(self) -> int:
        return self.vocab["E"]