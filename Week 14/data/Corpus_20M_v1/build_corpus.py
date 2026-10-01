"""Frozen tokenizer adapter extracted unchanged from the original corpus builder."""
from pathlib import Path
import sys, re, functools
import numpy as np
ROOT=Path(__file__).resolve().parent
UNITS = re.compile(r"\s+|[\w\u200c\u200d]+|[^\w\s]+", re.UNICODE)

class FastTokenizer:
    """Cache identical pretoken merges; preserve the frozen tokenizer's IDs."""
    def __init__(self):
        sys.path.insert(0, str(ROOT / "vendor"))
        from era6.tokenizer_candidates import load_lossless_tokenizer
        path = ROOT / "baseline/artifacts/tokenizer_v2/tokenizer.json"
        self.reference = load_lossless_tokenizer(path)
        self.specials = self.reference.special_token_ids
        self.pieces = self.reference._payloads
        self.hash = self.reference.tokenizer_hash
        assert self.reference._initialization == "bytes"
        assert not self.reference._initial_symbol_ids
        self.pattern = re.compile("|".join(re.escape(x) for x in sorted(self.specials, key=lambda x: (-len(x), x))))
        self.unit = functools.lru_cache(maxsize=200_000)(self._unit)

    def _unit(self, s):
        ids = self.reference._byte_ids
        symbols = [(ids[b], i, i + 1) for i, b in enumerate(s.encode())]
        return tuple(x[0] for x in self.reference._apply_merges(symbols))

    def encode(self, text):
        out = [self.specials["<bos>"]]
        pos = 0
        for match in self.pattern.finditer(text):
            for m in UNITS.finditer(text, pos, match.start()):
                out.extend(self.unit(m.group()))
            out.append(self.specials[match.group()])
            pos = match.end()
        for m in UNITS.finditer(text, pos):
            out.extend(self.unit(m.group()))
        out.append(self.specials["<eos>"])
        return np.asarray(out, dtype="<u2")

    def verify(self, text, ids, parity=False):
        decoded = b"".join(self.pieces[int(i)] for i in ids[1:-1]).decode()
        assert decoded == text, "Tokenizer round trip failed"
        if parity:
            expected = self.reference.encode(text, add_bos=True, add_eos=True)
            assert ids.tolist() == expected, "Cached tokenizer differs from original"

