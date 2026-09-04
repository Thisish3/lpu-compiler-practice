"""High-level "instruction blocks" - the API a model/kernel author calls.

Each function takes a Program and symbolic input names, emits Instr
objects into it, and returns the symbolic name(s) of its outputs. This
mirrors HyperDex's described API where developers compose blocks like
`input_load`, `token_embed`, `decoder`, or write a custom block for a
new operator - never touching registers or the RISC ISA directly.
"""

from ir import Program


def linear(prog: Program, x: str, w: str, hint: str = "lin") -> str:
    """y = x @ w"""
    return prog.emit("MATMUL", x, w, hint=hint)


def add(prog: Program, a: str, b: str, hint: str = "add") -> str:
    return prog.emit("ADD", a, b, hint=hint)


def relu(prog: Program, x: str, hint: str = "relu") -> str:
    return prog.emit("RELU", x, hint=hint)


def mlp_block(prog: Program, x: str, w1: str, w2: str) -> str:
    """A tiny feed-forward block: relu(x @ w1) @ w2 — stands in for one
    of HyperDex's higher-level blocks like `decoder`."""
    h = linear(prog, x, w1, hint="ffn_h")
    h = relu(prog, h, hint="ffn_act")
    return linear(prog, h, w2, hint="ffn_out")


def group_quant_block(prog: Program, x: str, group_size: int) -> tuple[str, str]:
    """Custom instruction block for the same absmax group-quant op we
    built as a Triton kernel in vllm-kernel-practice/. Here it's
    expressed as a chain of instruction-block calls instead of a Triton
    kernel body - same algorithm, different target (LPU vs GPU)."""
    absmax = prog.emit("REDUCE_ABS_MAX", x, hint="absmax", group_size=group_size)
    scale = prog.emit("DIV_SCALAR", absmax, hint="scale", value=127.0)
    scaled = prog.emit("DIV", x, scale, hint="scaled")
    clamped = prog.emit("CLAMP", scaled, hint="clamped", lo=-127, hi=127)
    q = prog.emit("CAST_INT8", clamped, hint="q")
    return q, scale