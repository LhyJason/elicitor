"""Questions the agent can ask (SPEC §7).

M1 has only open questions on one slot; choice and yes/no questions arrive with EIG in M3.
"""

from dataclasses import dataclass

OPEN = "open"


@dataclass(frozen=True)
class Question:
    slot: str
    kind: str = OPEN

    def to_dict(self) -> dict[str, str]:
        return {"slot": self.slot, "kind": self.kind}
