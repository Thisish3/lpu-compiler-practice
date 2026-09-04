"""A tiny linear-scan register allocator.

This is the part HyperDex's paper describes as automatic - "register
allocator tracks lifetime of every variable and automatically
allocates/deallocates, developer doesn't manage it manually." A Compute
Library Engineer writes at the ir.Program / blocks.py level; this file
is the layer *below* that they normally don't touch, implemented here
just to see concretely what it's doing.
"""

from dataclasses import dataclass

from ir import Instr, Program


class RegisterAllocationError(Exception):
    pass


@dataclass
class Assembled:
    reg_of: dict[str, int]
    num_registers_used: int
    instrs: list[Instr]

    def render(self) -> str:
        lines = []
        for i, instr in enumerate(self.instrs):
            args = ", ".join(f"r{self.reg_of[a]}" for a in instr.args)
            meta = f" {instr.meta}" if instr.meta else ""
            lines.append(
                f"{i:>3}: r{self.reg_of[instr.dst]} = {instr.op}({args}){meta}"
            )
        return "\n".join(lines)


def _compute_last_use(instrs: list[Instr], keep_alive: set[str]) -> dict[str, int]:
    """Index of the last instruction that reads each name.

    Names in `keep_alive` (the program's declared outputs) are pinned to
    "alive through the end" even if their natural last use is earlier -
    a caller still needs to read them after compilation finishes.
    """
    last_use: dict[str, int] = {}
    for i, instr in enumerate(instrs):
        for a in instr.args:
            last_use[a] = i
        last_use.setdefault(instr.dst, i)
    for name in keep_alive:
        last_use[name] = len(instrs)
    return last_use


def allocate_registers(
    prog: Program,
    inputs: list[str],
    outputs: list[str],
    num_registers: int,
) -> Assembled:
    """Assign a physical register (0..num_registers-1) to every symbolic
    name, freeing and reusing registers as soon as a value's last use has
    passed. No spilling: raises RegisterAllocationError if the pool runs
    out - a real compiler would spill the least-soon-needed live value to
    memory and reload it later instead of giving up.
    """
    instrs = prog.instrs
    last_use = _compute_last_use(instrs, set(outputs))

    free_regs = list(range(num_registers))
    reg_of: dict[str, int] = {}
    in_use: set[int] = set()
    peak = 0

    def alloc(name: str) -> int:
        nonlocal peak
        if not free_regs:
            raise RegisterAllocationError(
                f"'{name}' 할당 실패 - 레지스터 풀(={num_registers}) 소진. "
                "실제 컴파일러라면 여기서 가장 늦게 다시 쓰일 값을 메모리로 "
                "spill하고 나중에 reload하지만, 이 장난감 버전은 spill을 안 함."
            )
        reg = free_regs.pop()
        reg_of[name] = reg
        in_use.add(reg)
        peak = max(peak, len(in_use))
        return reg

    def free_if_dead(name: str, at: int) -> None:
        if name in reg_of and last_use.get(name) == at and name not in outputs:
            r = reg_of[name]
            if r in in_use:
                in_use.discard(r)
                free_regs.append(r)

    # external inputs are live from the very start
    for name in inputs:
        alloc(name)

    for i, instr in enumerate(instrs):
        # free operand registers that die exactly here *before* allocating
        # dst, so dst can immediately reuse a register an operand just
        # vacated (a real allocator does this too).
        for name in instr.args:
            free_if_dead(name, i)

        alloc(instr.dst)

        # a value that's produced but never read again (and isn't a
        # program output) can be freed immediately.
        free_if_dead(instr.dst, i)

    return Assembled(reg_of=reg_of, num_registers_used=peak, instrs=instrs)