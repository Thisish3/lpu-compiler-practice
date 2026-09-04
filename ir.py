"""Toy IR for an instruction-block style DSL.

Modeled after HyperAccel's LPU paper (arXiv:2408.07326) description of
HyperDex: developers compose named "instruction blocks" (input_load,
decoder, custom ops, ...) using symbolic tensor names, and a compiler
pass later assigns physical registers and emits a binary program. This
file is only the *symbolic* representation - no register allocation yet.
"""

from dataclasses import dataclass, field


@dataclass
class Instr:
    op: str
    dst: str
    args: list[str]
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        arg_str = ", ".join(self.args)
        meta_str = f" {self.meta}" if self.meta else ""
        return f"{self.dst} = {self.op}({arg_str}){meta_str}"


class Program:
    """A sequence of instructions over symbolic (unlimited) names.

    This is the level a "compute library engineer" writes at - like
    writing a Triton kernel with tl.load/tl.max instead of hand-placing
    registers. Analogous to HyperDex's Python instruction-block API.
    """

    def __init__(self):
        self.instrs: list[Instr] = []
        self._counter = 0

    def fresh(self, hint: str) -> str:
        self._counter += 1
        return f"{hint}_{self._counter}"

    def emit(self, op: str, *args: str, hint: str = "t", **meta) -> str:
        dst = self.fresh(hint)
        self.instrs.append(Instr(op, dst, list(args), meta))
        return dst

    def __repr__(self) -> str:
        return "\n".join(f"{i:>3}: {instr}" for i, instr in enumerate(self.instrs))
