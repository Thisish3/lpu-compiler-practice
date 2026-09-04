# LPU 스타일 instruction-block 컴파일러 미니 시뮬레이터

HyperAccel의 LPU 논문([arXiv:2408.07326](https://arxiv.org/abs/2408.07326))이 설명하는
HyperDex 소프트웨어 스택 - "개발자는 Python API로 instruction block을 조합하고,
컴파일러가 레지스터 할당/명령어 그룹화를 자동으로 해서 바이너리를 만든다" - 을
직접 손으로 구현해서 그 경계선이 정확히 어디인지 체감해보는 연습 프로젝트입니다.

## 구조

```
ir.py       - 심볼릭 IR (Instr, Program). 레지스터 개념이 아직 없음.
blocks.py   - "instruction block" API. mlp_block, group_quant_block처럼
              개발자가 실제로 호출하는 상위 레벨 함수들.
compiler.py - 레지스터 할당기 (linear-scan). 개발자가 안 건드리는 영역을
              직접 구현해본 것.
main.py     - 위 셋을 이어서 심볼릭 프로그램 -> "바이너리"(레지스터 배정된
              명령어 리스트)까지 컴파일하는 데모.
```

## 실행

```bash
python3 main.py
```

(순수 Python, GPU/CUDA 불필요)

## 무엇을 보여주는가

1. **1단계 출력**: `x_1`, `ffn_h_1` 같은 심볼릭 이름으로 된 명령어 시퀀스.
   이게 Compute Library Engineer가 실제로 짜는 레벨 — Triton 커널을 `tl.load`/`tl.max`로
   짜는 것과 같은 추상화 수준입니다.
2. **2단계 출력**: 레지스터 6개로 컴파일한 결과. 심볼릭 이름이 `r0`~`r5`로 바뀌고,
   죽은 값의 레지스터가 즉시 재사용됩니다 (`r4 = MATMUL(r5, r4)`처럼 입력 레지스터
   하나를 결과가 그대로 덮어쓰는 것도 실제 컴파일러/어셈블리에서 흔한 최적화).
   레지스터 6개를 줬지만 실제로 동시에 필요했던 건 3개뿐이었다는 것도 출력됩니다.
3. **3단계**: 레지스터를 2개만 주면 컴파일이 실패합니다 — 입력 3개(`x`, `w1`, `w2`)가
   동시에 살아있어야 하는데 자리가 없어서. 실제 컴파일러라면 이 시점에 값을 메모리로
   spill했을 텐데, 이 장난감 버전은 spill을 구현 안 해서 그냥 에러를 냅니다.

## `group_quant_block`

`blocks.py`의 `group_quant_block`은 [vllm-kernel-practice](https://github.com/Thisish3/vllm-kernel-practice)에서
Triton 커널로 직접 짰던 것과 **완전히 같은 absmax group quantization 알고리즘**을,
이번엔 "instruction block을 조합하는" 방식으로 표현한 것입니다. 같은 알고리즘이
타겟(GPU/Triton vs LPU/instruction-block)에 따라 어떻게 다르게 표현되는지 비교해보면
좋습니다.

## 이 연습에서 확인한 것

HyperDex 논문 원문: "레지스터 할당자가 모든 변수의 생명주기를 추적하고 자동으로
할당/해제하여, 개발자가 수동으로 관리할 필요가 없음."

→ 즉 Compute Library Engineer가 짜는 코드는 `blocks.py` 레벨까지고,
`compiler.py`가 하는 일(레지스터 생명주기 추적, 재사용, 실패 시 spill)은
별도 컴파일러 팀의 영역일 가능성이 높습니다. 이 구분선을 실제로 코드로
만들어보니 어디까지가 "라이브러리를 짜는 일"이고 어디부터가 "컴파일러를
만드는 일"인지 훨씬 명확해졌습니다.

## 한계 / 실제 컴파일러와 다른 점

- **Spilling 없음**: 레지스터가 부족하면 그냥 에러. 실제 컴파일러는 가장 늦게
  다시 쓰일 값을 골라(Belady's algorithm 등) 메모리로 내보내고 나중에 reload함.
- **Instruction chaining/grouping 없음**: 논문이 언급한 "명령어 그룹화"(아마 여러
  명령어를 하나의 VLIW 묶음이나 파이프라인 단계로 합치는 것)는 구현 안 함.
- **메모리 계층 모델링 없음**: 실제 LPU는 DRAM 대역폭 극대화가 핵심인데, 이 시뮬레이터는
  레지스터 하나만 다루고 메모리 접근 비용은 전혀 모델링하지 않음.