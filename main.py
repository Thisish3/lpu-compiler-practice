"""Demo: instruction-block program -> register-allocated "binary".

Mirrors the pipeline described in HyperAccel's LPU paper:
  developer's Python API calls (blocks.py)
    -> symbolic instruction program (ir.Program)
    -> compiler: register allocation (compiler.py)
    -> "binary" (physical-register instruction list)
"""

from blocks import group_quant_block, mlp_block
from compiler import RegisterAllocationError, allocate_registers
from ir import Program


def build_demo_program(prog: Program) -> tuple[list[str], list[str]]:
    """A tiny feed-forward block whose output gets group-quantized -
    same absmax quantization we built as a Triton kernel earlier, just
    expressed here as instruction blocks instead."""
    x, w1, w2 = "x", "w1", "w2"

    h = mlp_block(prog, x, w1, w2)
    q, scale = group_quant_block(prog, h, group_size=64)

    return [x, w1, w2], [q, scale]


def main():
    prog = Program()
    inputs, outputs = build_demo_program(prog)

    print("=== 1. instruction block 레벨 (심볼릭 이름, 레지스터 없음) ===")
    print(f"inputs={inputs}  outputs={outputs}")
    print(prog)
    print()

    print("=== 2. 레지스터 6개로 컴파일 ===")
    asm = allocate_registers(prog, inputs, outputs, num_registers=6)
    print(asm.render())
    print(f"\n실제 동시에 쓰인 최대 레지스터 수: {asm.num_registers_used} / 6")
    print()

    print("=== 3. 레지스터를 2개만 주면 (입력 3개도 동시에 못 들어감) ===")
    try:
        allocate_registers(prog, inputs, outputs, num_registers=2)
    except RegisterAllocationError as e:
        print("컴파일 실패 (의도된 결과):")
        print(" ", e)


if __name__ == "__main__":
    main()