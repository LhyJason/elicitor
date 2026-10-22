"""Questions the agent can ask (SPEC §7).

Three types, all about one slot:

- open: "what would you like?"; the outcomes are the slot's values.
- choice: "A, B or C?" between 2 or 3 concrete values; the outcomes are the offered values plus
  OTHER (none of them, or no preference).
- yes/no: "would A suit you?" about a set of values (one concrete value, or the first bins of an
  ordinal slot); the outcomes are YES and NO.

`noise_free_outcome` is the answer a user with a given true value gives when they answer
reliably; the noise model on top of it lives in `answer_model`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from elicitor.schema import CATEGORICAL, ORDINAL, Slot

OPEN = "open"
CHOICE = "choice"
YESNO = "yesno"
KINDS = (OPEN, CHOICE, YESNO)

OTHER = "other"
YES = "yes"
NO = "no"


@dataclass(frozen=True)
class Question:
    slot: str
    kind: str = OPEN
    values: tuple[str, ...] = ()  # choice: the offered values; yes/no: the values meaning "yes"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown question kind {self.kind!r}")
        if (self.kind == OPEN) != (not self.values):
            raise ValueError("open questions take no values; choice and yes/no questions do")

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"slot": self.slot, "kind": self.kind}
        if self.values:
            out["values"] = list(self.values)
        return out


def outcomes(slot: Slot, question: Question) -> tuple[str, ...]:
    if question.kind == OPEN:
        return slot.value_names
    if question.kind == CHOICE:
        return (*question.values, OTHER)
    return (YES, NO)


def noise_free_outcome(question: Question, value: str) -> str:
    if question.kind == OPEN:
        return value
    if question.kind == CHOICE:
        return value if value in question.values else OTHER
    return YES if value in question.values else NO


def candidates(slot: Slot, kinds: tuple[str, ...] = KINDS) -> list[Question]:
    """Every question on one slot, in the order open, choice, yes/no (SPEC §7)."""
    out = []
    if OPEN in kinds:
        out.append(Question(slot.name))
    if CHOICE in kinds:
        concrete = slot.concrete_values
        for size in (2, 3):
            out += [Question(slot.name, CHOICE, c) for c in itertools.combinations(concrete, size)]
    if YESNO in kinds:
        if slot.kind == CATEGORICAL:
            out += [Question(slot.name, YESNO, (v,)) for v in slot.concrete_values]
        elif slot.kind == ORDINAL:
            # "Below edge i?": yes means one of the first i bins.
            names = slot.value_names
            out += [Question(slot.name, YESNO, names[:i]) for i in range(1, len(names))]
    return out
