# CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs

---

## 1. 핵심 주장과 주요 기여 (요약)

### 핵심 주장

Transformer 훈련 시스템에서 GEMM(행렬 곱셈)은 이미 고도로 최적화되어 있지만, Normalization, Activation, Residual 업데이트 등 주변의 **메모리 바운드 연산**들이 전체 학습 시간의 상당 부분을 차지한다. CODA는 이러한 연산들을 **GEMM 에필로그(epilogue)** 안으로 흡수함으로써, 중간 텐서의 글로벌 메모리 왕복을 제거하는 GPU 커널 추상화이다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **GEMM-Epilogue 추상화** | Transformer의 비-Attention 연산을 GEMM 에필로그 프로그램으로 재파라미터화 |
| **5가지 에필로그 프리미티브** | Elementwise/Pairwise Map, Vector 로드/저장, Tile 로드/저장, Tile Reduction, Stateful Transform |
| **Forward+Backward 통합** | 순전파와 역전파 모두 동일한 GEMM-epilogue 구조로 표현 가능함을 정리(Theorem 1)로 증명 |
| **LLM 보조 커널 작성** | Claude Code가 CODA 프리미티브를 조합하여 고성능 커널을 자동 생성 |
| **수치적 정확도 향상** | 재파라미터화된 에필로그가 표준 PyTorch 경로보다 낮은 수치 오차를 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

H100에서 LLaMA-3-style 1B 모델을 TorchTitan으로 학습할 때, **GEMM과 Attention 이외의 연산(Normalization, Activation, Residual 등)이 전체 GPU 시간의 상당 부분**을 차지한다(Figure 1). FP8/FP4 등 저정밀도 행렬 연산의 가속은 빠르게 발전하지만, 중간 텐서 materialization 비용은 이에 비례해 줄어들지 않는다.

기존 접근법의 문제:
- **고수준 프레임워크(PyTorch)**: Operator 경계 = Materialization 경계 → 융합 기회 소실
- **수작업 커널(vLLM, SGLang 등)**: 고성능이나 새로운 변환/역전파 확장이 어려움
- **컴파일러 시스템(TVM, Triton 등)**: 빠르게 진화하는 하드웨어에 범용 최적화 어려움

### 2.2 제안하는 방법

#### 핵심 수식: GEMM-Plus-Epilogue 형태

```math
\text{GEMM: } \boldsymbol{h} = \boldsymbol{x}\boldsymbol{W}, \quad \text{Epilogue: } \boldsymbol{y}[i,j] = \boldsymbol{f}[i,j](\boldsymbol{h}[i,j])
```

여기서 $[i,j]$는 출력 타일 인덱스이며, $\boldsymbol{f}[i,j]$는 해당 타일에만 작용하는 타일 로컬 함수이다.

#### 패턴 1: GEMM-Residual-RMSNorm-GEMM

Pre-normalized Transformer에서 반복되는 패턴:

$$\boldsymbol{y} = \text{RMSNorm}(\boldsymbol{x}\boldsymbol{W}_0 + \boldsymbol{z}, \boldsymbol{\gamma})\boldsymbol{W}_1 = \left(r\left(\boldsymbol{x}\boldsymbol{W}_0 + \boldsymbol{z}\right) \odot \boldsymbol{\gamma}\right)\boldsymbol{W}_1$$

여기서 $r = 1/\text{rms}(\boldsymbol{x}\boldsymbol{W}_0 + \boldsymbol{z})$는 행 단위 역 RMS 인수이다.

**핵심 대수적 조작**: $r$은 행 공유이므로 두 번째 GEMM과 교환 가능:

$$\boldsymbol{y} = \left(r\left(\boldsymbol{x}\boldsymbol{W}_0 + \boldsymbol{z}\right) \odot \boldsymbol{\gamma}\right)\boldsymbol{W}_1 = r\left(\left(\boldsymbol{x}\boldsymbol{W}_0 + \boldsymbol{z}\right) \odot \boldsymbol{\gamma}\right)\boldsymbol{W}_1$$

이를 구체적으로 분해하면:

$$\text{GEMM 1: } \boldsymbol{h}_0 = \boldsymbol{x}\boldsymbol{W}_0$$

$$\text{Epilogue 1: } \boldsymbol{h}_1[i,j] = \boldsymbol{h}_0[i,j] + \boldsymbol{z}[i,j]$$
$$\boldsymbol{h}_2[i,j] = \boldsymbol{h}_1[i,j] \odot \boldsymbol{\gamma}[j]$$
$$\hat{\boldsymbol{r}}[i,j] = \text{partialRMS}(\boldsymbol{h}_1[i,j])$$

$$\text{GEMM 2: } \boldsymbol{h}_3 = \boldsymbol{h}_2\boldsymbol{W}_1, \quad \text{Epilogue 2: } \boldsymbol{y}[i,j] = r[i]\boldsymbol{h}_3[i,j]$$

$$r = 1/\sqrt{\text{reduce}(\hat{\boldsymbol{r}}) + \epsilon}$$

#### 패턴 2: GEMM with Pairwise Activations (RoPE, SwiGLU)

$$\boldsymbol{h} = \boldsymbol{x}\boldsymbol{W}, \quad \boldsymbol{h}_a[i,j], \boldsymbol{h}_b[i,j] = \text{split}(\boldsymbol{h}[i,j]), \quad \boldsymbol{y}[i,j] = \boldsymbol{f}[i,j](\boldsymbol{h}_a[i,j], \boldsymbol{h}_b[i,j])$$

Hopper TensorCore의 에필로그에서 각 스레드가 인접한 레지스터 값 쌍을 보유하므로, RoPE나 SwiGLU 같은 쌍별 활성화를 레지스터 수준에서 직접 계산 가능하다.

#### 패턴 3: GEMM with Cross-Entropy Loss

토큰 $i$에 대한 손실:

$$\ell_i = -h_{i,y_i} + \log\sum_k \exp(h_{i,k})$$

타겟 로짓 선택과 log-sum-exp를 에필로그의 타일 로컬 통계로 누적하고, 소형 보조 reduction으로 병합한다.

#### 역전파 구조 (Theorem 1)

GEMM-with-epilogue 블록 시퀀스:

```math
\boldsymbol{h}_\ell = \boldsymbol{x}_{\ell-1}\boldsymbol{W}_\ell, \quad \boldsymbol{x}_\ell[i,j] = \boldsymbol{f}_\ell[i,j](\boldsymbol{h}_\ell[i,j]), \quad \ell = 1, \ldots, L-1
```

역전파도 동일한 구조를 보존한다:

$$\nabla_{\boldsymbol{x}_{\ell-1}}\mathcal{L} = \nabla_{\boldsymbol{h}_\ell}\mathcal{L}\boldsymbol{W}_\ell^\top$$
$$\nabla_{\boldsymbol{h}_{\ell-1}}\mathcal{L}[i,j] = \boldsymbol{g}_{\ell-1}[i,j]\left(\nabla_{\boldsymbol{x}_{\ell-1}}\mathcal{L}[i,j],\ \boldsymbol{h}_{\ell-1}[i,j]\right)$$
$$\nabla_{\boldsymbol{W}_\ell}\mathcal{L} = \boldsymbol{x}_{\ell-1}^\top \nabla_{\boldsymbol{h}_\ell}\mathcal{L}$$

**RMSNorm 역전파의 행 단위 통계 이동**: RMSNorm backward에서 필요한 행 단위 내적 $\boldsymbol{s}$:

$$\boldsymbol{s} = \frac{1}{d}\text{sum}_{\text{cols}}(\nabla_{\boldsymbol{h}_2}\mathcal{L} \odot \boldsymbol{h}_2)$$

$\nabla_{\boldsymbol{h}\_2}\mathcal{L} = \nabla_{\boldsymbol{y}}\mathcal{L}\boldsymbol{W}_1^\top$와 $\boldsymbol{y} = \boldsymbol{h}_2\boldsymbol{W}_1$을 이용하면:

$$\boldsymbol{s} = \frac{1}{d}\text{sum}_{\text{output}}(\nabla_{\boldsymbol{y}}\mathcal{L} \odot \boldsymbol{y})$$

이 항등식 덕분에 RMSNorm backward 통계를 인접 GEMM 에필로그에서 누적할 수 있다.

### 2.3 모델 구조 (CODA 시스템 구조)

```
┌─────────────────────────────────────────────┐
│              GEMM Mainloop (고정)             │
│   (CuTeDSL 기반, Hopper TensorCore 최적화)    │
└──────────────────────┬──────────────────────┘
                       │ 출력 타일 (온칩)
┌──────────────────────▼──────────────────────┐
│           Epilogue (프로그래머블)              │
│  ① Elementwise/Pairwise Maps               │
│     (Residual, Activation, RoPE, SwiGLU)   │
│  ② Vector(Rank-1) Loads/Stores            │
│     (RMSNorm weight 브로드캐스트)            │
│  ③ Tile(Rank-2) Loads/Stores              │
│     (Residual stream, saved activations)   │
│  ④ Tile Reductions                        │
│     (Partial row/col reductions)           │
│  ⑤ Stateful Transforms                    │
│     (Online log-sum-exp, cross-entropy)    │
└──────────────────────┬──────────────────────┘
                       │ 경량 보조 Reduction
                       ▼
                  글로벌 메모리 저장
```

**구현 기반**: NVIDIA CuTeDSL (Python 수준 커널 작성 + 저수준 레이아웃 제어), CUTLASS의 Epilogue Visitor Tree 개념 확장.

### 2.4 성능 향상

실험은 단일 H100 GPU에서 진행. 기준선: cuBLAS + `torch.compile`.

**커널 수준 속도 향상 (Figure 8, 10)**:

| 커널 | 히든 차원 | CODA(LLM) 속도향상 |
|---|---|---|
| GEMM-Residual-RMS-GEMM | d=2048~8192 | ~1.05–1.15× |
| GEMM+RoPE | d=4096~8192 | ~1.1–1.2× |
| GEMM+SwiGLU | d=4096~8192 | ~1.0–1.05× |
| GEMM-Residual-PartialRMS-GEMM (Backward) | d=4096~8192 | ~1.4–1.8× |
| GEMM-RMS-SwiGLU (Backward) | d=4096~8192 | ~1.3–1.6× |

**블록 수준 속도 향상 (Figure 11)**:

- Layer Forward: d=2048에서 약 1.1× 향상
- Layer Forward+Backward: 전반적으로 1.0–1.1× 수준

**수치 정확도 (Figure 6)**:
- CODA의 에러 비율이 표준 PyTorch 대비 낮음(0.5–0.8 범위). 더 정확한 GEMM 메인루프(QuACK 기반)가 에필로그 재파라미터화와 결합되어 수치 오차를 추가 감소.

### 2.5 한계

1. **아키텍처 의존성**: 현재 표준 Transformer++ 구조(LLaMA 계열) 중심. MoE, SSM(Mamba), Linear Attention 등 비표준 아키텍처로의 확장은 미완.
2. **단일 GPU 범위**: 분산 실행(Tensor Parallelism, Pipeline Parallelism) 미지원.
3. **모듈 경계 모호화**: 재파라미터화가 모듈 경계를 흐려 프레임워크 수준 추상화와의 통합을 어렵게 함.
4. **타일 로컬 제약**: 에필로그는 타일 로컬 연산에 제한됨. 전역 통신이 필요한 연산은 별도 패스 필요.
5. **CuTeDSL 노출 부족**: 현재 LLM 모델들이 CuTeDSL 이디엄에 덜 익숙해 LLM 보조 작성 품질이 가변적.

---

## 3. 일반화 성능 향상 가능성

이 섹션은 **모델의 학습 일반화(training generalization)** 측면에서 CODA가 미칠 수 있는 영향을 분석한다. 논문 자체는 시스템 효율성에 초점을 맞추고 있으나, 다음 메커니즘을 통해 일반화에 간접적으로 기여할 수 있다.

### 3.1 수치 정밀도 향상을 통한 학습 안정성

$$\text{Error}_{\text{CODA}} < \text{Error}_{\text{PyTorch}} \approx \text{Error}_{\text{QuACK}}$$

Figure 6에서 CODA의 수치 오차 비율이 표준 PyTorch보다 낮음이 확인된다. BF16/FP8 학습에서 누적 수치 오차는 그래디언트 노이즈와 유사하게 작용하여 최적화 경로에 영향을 준다. 더 정확한 중간 계산은:
- **그래디언트의 분산 감소** → 더 안정적인 파라미터 업데이트
- **RMSNorm 통계의 정확도 향상** → 정규화 품질 개선 → 일반화에 유리

특히 역전파 RMSNorm 통계 이동 공식:

$$\boldsymbol{s} = \frac{1}{d}\text{sum}_{\text{output}}(\nabla_{\boldsymbol{y}}\mathcal{L} \odot \boldsymbol{y})$$

이 등가 변환이 activation-sized 텐서 재독시 발생하는 반올림 오차를 줄여준다.

### 3.2 더 큰 배치/긴 시퀀스 학습 가능성

메모리 바운드 연산 감소는 단순 속도 향상을 넘어:
- **같은 메모리 예산으로 더 큰 배치 크기** 사용 가능 → 배치 크기가 일반화에 영향을 주는 경우(Sharpness-Aware Minimization 등) 유리
- **더 긴 시퀀스 처리** 가능 → 장거리 의존성 학습 개선

### 3.3 FP8/FP4 정밀도 학습 지원

논문은 FP8 학습에서 "Others" 카테고리(비-GEMM 연산)의 비율이 BF16보다 더 크다고 지적한다(Figure 1). CODA는 이 병목을 에필로그 융합으로 완화하여 **저정밀도 학습의 실용성을 높인다**. 저정밀도 학습의 성공적 적용은 더 큰 모델 훈련을 가능하게 하며, 이는 일반적으로 일반화 성능 향상과 연관된다.

### 3.4 정규화 계산의 충실도

GEMM-RMSNorm-GEMM 패턴에서, 표준 구현은 RMSNorm을 별도 커널로 실행하면서 글로벌 메모리 왕복 시 정밀도 손실이 발생한다. CODA는 $r$ 인수를 온칩에서 계산 및 적용하므로:
- **RMSNorm의 정규화 효과가 더 충실하게 반영** → 특히 FP8 환경에서의 학습 안정성 개선
- **Cross-entropy 손실의 수치 안정성 향상** → log-sum-exp를 온칩에서 수행하여 overflow/underflow 감소

### 3.5 한계: 직접적 일반화 실험 부재

논문은 downstream task 성능이나 perplexity 등 **모델 일반화를 직접 측정하는 실험을 포함하지 않는다**. 위의 분석은 시스템 특성에서 추론한 간접적 효과이며, 실제 일반화 향상을 확인하려면 추가 실험이 필요하다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 관련 연구 맵

```
커널 최적화 접근법
├── DSL/라이브러리 기반
│   ├── Triton (2019/2021~) ─── 범용 타일 프로그래밍
│   ├── ThunderKittens (2024) ── 계층적 CUDA 추상화
│   ├── TileLang (2025) ──────── 조합 가능한 타일 모델
│   └── CuTeDSL (2023~) ──────── CUTLASS Python 바인딩
├── 컴파일러 기반
│   ├── TVM/Relax (2018~) ─────── 자동 End-to-End 최적화
│   ├── torch.compile (2024) ──── PyTorch 2.0 컴파일러
│   ├── TASO (2019) ───────────── 그래프 치환 자동 생성
│   └── Mirage (2025) ─────────── 다단계 텐서 슈퍼최적화
├── LLM 특화 라이브러리
│   ├── FlashAttention (2022~) ── Attention 전용 IO-aware
│   ├── Liger Kernels (2024) ──── Triton 기반 LLM 커널
│   ├── FlashInfer (2025) ──────── LLM 추론 특화
│   └── Cut Cross-Entropy (2025) ─ 대어휘 CE 손실 최적화
└── LLM 보조 커널 생성
    ├── KernelBench (2025) ─────── LLM 커널 작성 벤치마크
    ├── CUDA-LLM (2025) ────────── LLM CUDA 커널 생성
    └── CODA (2026) ────────────── GEMM-Epilogue 프레임워크
```

### 4.2 세부 비교

| 연구 | 연도 | 핵심 접근 | CODA 대비 차이점 |
|---|---|---|---|
| **FlashAttention** (Dao et al.) | 2022 | IO-Aware Attention 커널, Tiling으로 메모리 감소 | Attention에 특화, 비-Attention 연산 미포함 |
| **Liger Kernels** | 2024 | Triton 기반 LLM 학습 커널 모음 | 개별 연산 최적화, 교차 연산 에필로그 융합 없음 |
| **ThunderKittens** | 2024 | CUDA 커널을 계층적 타일 추상화로 단순화 | 범용 CUDA 추상화, Transformer 에필로그 특화 아님 |
| **TileLang** | 2025 | 조합 가능한 타일 프로그래밍 모델 | 범용 타일 언어, GEMM-epilogue 특화 패턴 없음 |
| **Cut Cross-Entropy** | 2025 (ICLR) | 대어휘 CE 손실을 GEMM 에필로그로 표현 | CE 손실에 한정; CODA는 이를 포함한 더 넓은 범위 |
| **Mirage** | 2025 | 다단계 텐서 슈퍼최적화기 | 자동 탐색 기반, 수작업 대수적 재파라미터화 없음 |
| **EVT (Epilogue Visitor Tree)** | 2024 | CUTLASS 에필로그 방문자 트리 형식화 | CODA의 기반 기술; CODA는 Transformer 특화로 확장 |
| **KernelBench** | 2025 | LLM이 GPU 커널 작성 가능 여부 평가 | 평가 프레임워크; CODA는 LLM 보조 작성을 실제 구현 |
| **CUDA-LLM** | 2025 | LLM이 효율적 CUDA 커널 작성 | 임의 CUDA 합성; CODA는 제약된 프리미티브 조합 |

### 4.3 CODA의 차별점

CODA는 **"대수적 재파라미터화 + 도메인 특화 에필로그 프리미티브 + LLM 보조 작성"**의 교차점에 위치한다:

1. **Cut Cross-Entropy**와 유사한 에필로그 융합 철학을 가지나, **전체 Transformer 블록(Forward+Backward)**으로 범위를 확장
2. **Triton/ThunderKittens**이 임의 커널 작성을 지원하는 반면, CODA는 **고성능 GEMM 메인루프를 고정**하고 에필로그만 프로그래밍하여 실수 가능성 감소
3. **torch.compile** 등 컴파일러 접근법이 자동 융합을 시도하지만, CODA는 **명시적 대수 변환**으로 컴파일러가 놓치는 교차-레이어 융합(예: RMSNorm 통계 이동)을 포착

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 향후 연구에 미치는 영향

#### 시스템 연구 측면

**① GEMM-Epilogue를 기본 단위로 삼는 새로운 컴파일러 설계**
- 기존 컴파일러가 개별 연산자를 최적화 단위로 삼는 반면, CODA는 "GEMM + 에필로그 프로그램"을 원자 단위로 제시한다. 향후 LLM 컴파일러들이 이 단위를 기본 IR(Intermediate Representation)로 채택할 가능성이 있다.

**② LLM 보조 커널 생성의 실용화**
- KernelBench 등이 평가 중인 "LLM이 GPU 커널을 쓸 수 있는가?" 문제에, CODA는 **"제약된 도메인에서는 이미 실용적"**이라는 긍정적 증거를 제공한다. 도메인 특화 프리미티브 세트 + 큐레이션된 예시 = LLM이 고성능 커널 작성 가능.

**③ 역전파 자동화의 새로운 경로**
- Theorem 1은 타일 로컬 에필로그가 역전파에서도 타일 로컬 구조를 보존함을 형식적으로 증명한다. 이는 **자동 미분 시스템이 에필로그 수준에서 동작**할 수 있는 이론적 기반을 제공한다.

**④ 저정밀도(FP8/FP4) 학습 인프라**
- 하드웨어가 저정밀도 GEMM을 가속할수록 비-GEMM 연산의 상대적 비용이 증가한다. CODA의 접근법은 이 추세에 대응하는 핵심 방법론이 될 수 있다.

#### 모델링 연구 측면

**⑤ 새로운 Transformer 변형 설계 시 에필로그 효율성 고려**
- 연구자들이 새로운 normalization이나 activation 함수를 제안할 때, "이 연산이 GEMM 에필로그로 표현 가능한가?"를 설계 기준으로 삼을 수 있다.

### 5.2 향후 연구 시 고려할 사항

#### 기술적 확장

**① 분산 실행으로의 확장**
- 현재 단일 GPU에 한정. Tensor Parallelism 환경에서 에필로그가 All-Reduce와 어떻게 상호작용하는지 연구 필요.
- Pipeline Parallelism에서 에필로그가 버블을 줄일 수 있는지 탐구 필요.

**② 다양한 아키텍처 지원**
- MoE(Mixture of Experts): 게이팅 메커니즘이 타일 로컬 연산인지 분석 필요.
- Linear Attention/SSM(Mamba): 순환 상태 업데이트가 에필로그로 표현 가능한지 검토.
- GQA(Grouped Query Attention): KV 캐시 재사용 패턴이 에필로그 구조에 적합한지 평가.

**③ 새로운 GPU 아키텍처 대응**
- 논문은 H100(Hopper) 기반. Blackwell(B100/B200)의 TMEM 기반 파이프라인에 최적화된 에필로그 설계 연구 필요.
- FP4 학습을 위한 에필로그 내 데이터 타입 변환 최적화.

#### 평가 및 검증

**④ 실제 모델 학습에서의 종단간 검증**
- 현재 커널/블록 수준 벤치마크만 존재. 전체 LLM 사전학습에서의 wall-clock time 감소, 수렴 품질, 최종 모델 성능 비교 필요.

**⑤ 일반화 성능에 대한 직접 실험**
- 수치 정밀도 향상이 실제 downstream 성능(perplexity, 벤치마크 점수)에 미치는 영향 측정.
- 특히 FP8 학습에서 CODA vs. 표준 학습의 모델 품질 비교.

**⑥ 자동화 수준 향상**
- 현재 LLM 보조 작성에 "경량 인간 감독" 필요. 완전 자동화(자동 재파라미터화 발견 + 에필로그 합성)를 위한 연구 필요.
- 재파라미터화 가능성을 자동으로 탐색하는 프로그램 합성 시스템 연구.

#### 소프트웨어 생태계

**⑦ 프레임워크 통합**
- 논문이 지적하듯 재파라미터화가 모듈 경계를 흐린다. PyTorch Autograd와 깔끔하게 통합되는 인터페이스 설계 연구 필요.
- `torch.compile`과의 공존 또는 통합 방안 탐구.

**⑧ 프리미티브 세트 확장**
- 현재 5가지 프리미티브로 표준 Transformer++를 거의 커버하지만, Flash Linear Attention의 청크별 누적, RMS-free 아키텍처 등 새로운 패턴에 대한 프리미티브 추가 연구 필요.

---

## 참고 자료

### 주요 논문 (직접 인용)

1. **Han Guo, Jack Zhang, Arjun Menon, Driss Guessous, Vijay Thakkar, Yoon Kim, Tri Dao. "CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs." arXiv:2605.19269v2, 2026.** (본 논문)

2. **Z. Chen, A. Kerr, R. Cai, J. Kosaian, H. Wu, Y. Ding, and Y. Xie. "EVT: Accelerating deep learning training with epilogue visitor tree." ASPLOS 2024, Volume 3, pp. 301–316.**

3. **E. Wijmans, B. Huval, A. Hertzberg, V. Koltun, and P. Krähenbühl. "Cut your losses in large-vocabulary language models." ICLR 2025.**

4. **A. Ivanov, N. Dryden, T. Ben-Nun, S. Li, and T. Hoefler. "Data movement is all you need: A case study on optimizing transformers." Proceedings of Machine Learning and Systems, 3:711–732, 2021.**

5. **V. Thakkar et al. "CUTLASS." NVIDIA, 2023. https://github.com/NVIDIA/cutlass**

6. **P.-L. Hsu et al. "Liger kernel: Efficient triton kernels for llm training." arXiv:2410.10989, 2024.**

7. **Z. Ye et al. "FlashInfer: Efficient and customizable attention engine for llm inference serving." MLSys 2025.**

8. **B. F. Spector et al. "ThunderKittens: Simple, fast, and adorable AI kernels." arXiv:2410.20399, 2024.**

9. **L. Wang et al. "TileLang: A composable tiled programming model for AI systems." arXiv:2504.17577, 2025.**

10. **M. Wu et al. "Mirage: A Multi-Level superoptimizer for tensor programs." OSDI 2025, pp. 21–38.**

11. **A. Ouyang et al. "KernelBench: Can LLMs write efficient GPU kernels?" arXiv:2502.10517, 2025.**

12. **W. Chen et al. "CUDA-LLM: LLMs can write efficient CUDA kernels." arXiv:2506.09092, 2025.**

13. **W. Liang et al. "TorchTitan: One-stop pytorch native solution for production ready LLM pre-training." arXiv:2410.06511, 2024.**

14. **J. Su et al. "RoFormer: Enhanced transformer with rotary position embedding." Neurocomputing, 568:127063, 2024.**

15. **A. Grattafiori et al. "The Llama 3 herd of models." arXiv:2407.21783, 2024.**

16. **코드 저장소: https://github.com/HanGuo97/coda-kernels**

17. **QuACK 커널: https://github.com/Dao-AILab/quack**
