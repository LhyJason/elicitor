"""Seeded random generators derived from a context (SPEC §11).

The same parts always give the same stream, on every platform and Python run (no use of the
salted built-in `hash`).
"""

import hashlib
import random


def derive_seed(*parts: object) -> int:
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def derive_rng(*parts: object) -> random.Random:
    return random.Random(derive_seed(*parts))
