# FairyFuse: Multiplication-Free LLM Inference on CPUs via Fused Ternary Kernels 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

FairyFuse는 **삼진수(Ternary) 가중치 $w \in \{-1, 0, +1\}$** 를 활용하여 CPU 상에서 부동소수점 곱셈(Floating-Point Multiplication)을 완전히 제거한 LLM 추론 시스템이다. 기존 시스템들이 저비트 양자화 후에도 여전히 역양자화(dequantization) 또는 룩업 테이블(LUT)을 통해 곱셈을 수행하는 것과 달리, FairyFuse는 AVX-512 마스크 기반 덧셈/뺄셈만으로 내적 루프를 구성한다.

### 주요 기여 3가지

| 기여 | 내용 |
|------|------|
| **① 곱셈-없는 GEMV 커널** | x86 AVX-512 상에서 마스크 덧셈/뺄셈만을 사용하는 첫 번째 삼진수 GEMV 커널 구현 |
| **② 8-GEMV 융합 커널** | 복소수 광선형(Widely-Linear) 레이어의 8개 서브-GEMV를 단일 SIMD 루프로 통합 (1.55× 가속) |
| **③ CPU 우위 입증** | Roofline 분석을 통해 삼진수 패킹이 GPU가 아닌 CPU에서 구조적으로 유리함을 증명 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**문제 1: CPU 메모리 대역폭 병목**

자동회귀 디코딩(Autoregressive Decoding)에서 각 토큰 생성 시 모든 가중치를 최소 1회 스트리밍해야 하므로, 처리량은 산술 연산 한계가 아닌 **메모리 대역폭**에 의해 결정된다. 약 200 GB/s 대역폭의 CPU에서 이 병목은 특히 심각하다.

**문제 2: 기존 시스템의 삼진수 구조 미활용**

- **llama.cpp**: Q2_K, Q4_K_M 등에서 역양자화 후 FP16 FMA 실행 → 저비트 인코딩이 저장 최적화에만 사용됨
- **T-MAC / BitNet.cpp**: VPSHUFB 기반 LUT 조회 → 간접 메모리 접근 및 L1/L2 캐시 압박 발생
- **Fairy2i 참조 구현**: PyTorch `F.linear` 연산자 통해 표준 곱셈-누산 루틴 사용 → 명령어 수준에서 삼진수 구조 미활용

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 삼진수 가중치의 곱셈-없는 분해

삼진수 가중치 $w \in \{-1, 0, +1\}$와 활성화값 $a \in \mathbb{R}$에 대해:

$$w \cdot a = \begin{cases} a & \text{if } w = +1 \\ -a & \text{if } w = -1 \\ 0 & \text{if } w = 0 \end{cases}$$

**2비트 패킹 방식**: 16개의 삼진수 가중치를 하나의 `uint32`에 인코딩

$$\text{packed} = \sum_{k=0}^{15} \left[ \mathbb{1}[t_k = +1] \cdot 2^{2k+1} + \mathbb{1}[t_k = -1] \cdot 2^{2k} \right] $$

- $(1, 0)$: $+1$ (add)
- $(0, 1)$: $-1$ (subtract)
- $(0, 0)$: $0$ (no-op)

#### (B) 병렬 비트 디코드 (BMI2 `_pext_u32`)

$$k_{\text{pos}} = \texttt{pextu32}(p,\ M_{\text{pos}}), \quad M_{\text{pos}} = \texttt{0xAAAAAAAAu} $$

$$k_{\text{neg}} = \texttt{pextu32}(p,\ M_{\text{neg}}), \quad M_{\text{neg}} = \texttt{0x55555555u} $$

#### (C) 마스크 기반 누산 (AVX-512)

16-wide FP32 청크 $\mathbf{x} \in \mathbb{R}^{16}$에 대해:

$$\text{acc} \leftarrow \texttt{\mm512\mask\add\ps}(\text{acc},\ k_{\text{pos}},\ \text{acc},\ X) $$

$$\text{acc} \leftarrow \texttt{\mm512\mask\sub\ps}(\text{acc},\ k_{\text{neg}},\ \text{acc},\ X) $$

**→ 내부 루프 내 부동소수점 곱셈 명령어 수 = 0**

#### (D) 복소수 광선형 레이어 수식

복소수 가중치 $\mathbf{U}, \mathbf{W} \in \mathbb{C}^{n \times m}$와 활성화 $\mathbf{x} \in \mathbb{C}^m$에 대해:

$$\mathbf{y} = \mathbf{U}\mathbf{x} + \mathbf{W}\bar{\mathbf{x}} $$

실수/허수 성분으로 분해하면 **8개의 실수 GEMV**로 전개:

$$y_{\text{re}} = s^U_{\text{re}} \mathbf{U}_{\text{re}}\mathbf{x}_{\text{re}} - s^U_{\text{im}} \mathbf{U}_{\text{im}}\mathbf{x}_{\text{im}} + s^W_{\text{re}} \mathbf{W}_{\text{re}}\mathbf{x}_{\text{re}} + s^W_{\text{im}} \mathbf{W}_{\text{im}}\mathbf{x}_{\text{im}} $$

$$y_{\text{im}} = s^U_{\text{re}} \mathbf{U}_{\text{re}}\mathbf{x}_{\text{im}} + s^U_{\text{im}} \mathbf{U}_{\text{im}}\mathbf{x}_{\text{re}} + s^W_{\text{re}} \mathbf{W}_{\text{re}}\mathbf{x}_{\text{im}} - s^W_{\text{im}} \mathbf{W}_{\text{im}}\mathbf{x}_{\text{re}} $$

모든 가중치 블록이 삼진수이므로 $O(nm)$ 지배 연산은 곱셈-없음. 스케일 적용 $O(n)$만 부동소수점 곱셈을 유지하지만, 이는 점근적으로 무시 가능.

---

### 2.3 모델 구조 (FairyFuse 시스템 구조)

```
[오프라인 단계]
FP16 가중치 (13.5 GB)
       ↓ PhaseQuant QAT (Fairy2i)
삼진수 가중치 {±1, ±i} → 2비트 패킹 (3.3 GB, 16× 압축)

[온라인 추론 단계 (토큰 당)]
입력 토큰
       ↓
× 32 Transformer 레이어:
  ① RMSNorm (FP32)
  ② FairyFuse 융합 8-GEMV (곱셈-없는, AVX-512)
     - Q/K/V 투영: 285 μs (30.2%)
     - O 투영: 95 μs (10.1%)
     - Gate+Up 투영: 380 μs (40.3%)
     - Down 투영: 95 μs (10.1%)
  ③ SiLU 활성화 (FP32)
       ↓
출력 토큰 (32.4 tok/s)
```

**융합 커널의 4가지 핵심 최적화**:

| 최적화 | 설명 | 기여도 |
|--------|------|--------|
| **O1: 마스크 재사용** | 디코딩된 마스크 쌍을 실수/허수 경로 모두에 적용 | ~15% |
| **O2: 입력 벡터 재사용** | $\mathbf{x}\_{\text{re}}, \mathbf{x}_{\text{im}}$을 청크당 1회만 로드 | ~20% |
| **O3: 부호 교환 트릭** | `neg_x_im = -x_im` 사전 계산으로 켤레 항 처리 | ~5% |
| **O4: 레지스터 상주 누산기** | 8개 부분합을 최종 수평 축소 전까지 AVX-512 레지스터에 유지 | ~10% |

---

### 2.4 성능 향상

#### 커널 수준 성능 (DRAM-cold)

| 행렬 크기 | FP32 1t (μs) | Ternary 1t (μs) | Ternary 48t (μs) | 1t 가속 | 48t 가속 |
|-----------|-------------|-----------------|------------------|---------|----------|
| 4096×4096 | 12,550 | 2,410 | 424 | 5.2× | **29.6×** |
| 11008×4096 | 42,730 | 6,480 | 786 | 6.6× | **54.4×** |
| 4096×11008 | 13,220 | 6,500 | 447 | 2.0× | **29.6×** |

#### 엔드-투-엔드 성능 및 품질 (LLaMA-2-7B)

| 방법 | 처리량 (tok/s) | 메모리 (GB) | WikiText-2 PPL | 다운스트림 정확도 |
|------|--------------|------------|---------------|-----------------|
| FP16 | 8.24 | 13.5 | 5.47 | 67.3% |
| llama.cpp Q4_K_M | 26.15 | 4.1 | 5.68 | 65.1% |
| **FairyFuse (ours)** | **32.43** | **3.3** | **5.52** | **66.0%** |
| llama.cpp Q2_K | 20.10 | 2.8 | 7.82 | 56.6% |

#### 어셈블리 검증

| 명령어 | 개수 | 역할 |
|--------|------|------|
| `pext` | 8 | 가중치 마스크 디코드 |
| `vaddps` (마스크) | 8 | 마스크 덧셈 |
| `vsubps` (마스크) | 8 | 마스크 뺄셈 |
| `vmulps` / `vfmadd` | **0** | **곱셈 없음** |

#### Roofline 분석: CPU vs GPU 비대칭성

**산술 강도(Arithmetic Intensity, AI) 비교**:

$$\text{AI}_{\text{FP32 dense}} = 0.25 \ \text{FLOP/byte}$$

$$\text{AI}_{\text{Ternary single}} = 4.0 \ \text{OP/byte}$$

$$\text{AI}_{\text{Ternary fused (8-GEMV)}} = 8.0 \ \text{OP/byte}$$

| 플랫폼 | 피크 대역폭 | 능선점 (Ridge Point) | Ternary AI=8.0 위치 |
|--------|-----------|---------------------|---------------------|
| CPU Xeon 8558P | 200 GB/s | 13.5 FLOP/byte | 능선점에 근접 → **29.6× 가속** |
| GPU H200 NVL | 4,800 GB/s | 27.9 FLOP/byte | 능선점 미달 → **130× 느려짐** |

GPU는 `_pext_u32`에 해당하는 단일 사이클 명령어가 없어 구조적으로 삼진수 추론에 불리하다.

---

### 2.5 한계

1. **단일 모델/ISA**: LLaMA-2-7B 및 x86 AVX-512만 평가; ARM NEON으로 확장 필요
2. **Fairy2i 의존성**: Fairy2i 복소수 양자화에만 호환; 타 삼진수 레시피 미지원
3. **배치 GEMM 미지원**: 프롬프트 처리 및 투기적 디코딩을 위한 배치 GEMM 미구현
4. **대형 모델 미검증**: 13B, 70B 모델에서의 최적 스레드-행 매핑 변화 가능성
5. **NUMA 민감성**: 크로스-소켓 실행 시 ~10% 처리량 저하

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Fairy2i의 복소수 표현이 일반화에 기여하는 이유

FairyFuse의 품질 보존은 기반 모델인 **Fairy2i의 PhaseQuant** 양자화 인식 훈련(QAT)에서 기인한다. 복소수 삼진수 표현 $\{+1, -1, +i, -i\}$는 실수 공간의 $\{-1, 0, +1\}$로 분해되며, 이 **위상(Phase) 구조**가 일반화에 유리한 이유는 다음과 같다:

**① 표현력의 풍부함**: 복소수 도메인에서의 가중치는 실수-삼진수 대비 위상 정보를 추가로 인코딩하여, 동일한 2비트 저장 예산 내에서 더 풍부한 표현 공간을 활용한다.

**② 양자화 오차의 분산**: 실수/허수 성분이 각각 독립적으로 삼진수화되므로, 양자화 오차가 복소수 공간 전체에 분산되어 특정 방향으로의 편향이 줄어든다.

**③ 스케일의 정밀한 조정**: 각 채널별 학습 가능한 스케일 $s^U_{\text{re}}, s^U_{\text{im}}, s^W_{\text{re}}, s^W_{\text{im}}$이 4개의 독립된 실수 GEMV 경로를 개별 보정하여, 단순 삼진수 대비 정보 손실을 최소화한다.

**④ 실험적 일반화 증거**:

| 태스크 | FP16 | FairyFuse | Q4_K_M | Q2_K |
|--------|------|-----------|--------|------|
| ARC-E | 75.4 | 74.2 | 73.8 | 62.1 |
| ARC-C | 43.5 | 41.8 | 41.2 | 33.4 |
| HellaSwag | 72.1 | 71.6 | 70.8 | 58.2 |
| PIQA | 79.0 | 78.3 | 78.0 | 72.4 |
| WinoGrande | 66.5 | 64.2 | 61.7 | 56.8 |
| **평균** | **67.3%** | **66.0%** | **65.1%** | **56.6%** |

FairyFuse는 FP16 대비 **-1.3 pp** 저하에 그치며, Q2_K의 **-10.7 pp** 저하와 극명히 대조된다. 특히 추론 능력이 필요한 ARC-Challenge에서 Q2_K는 -10.1 pp 하락하지만 FairyFuse는 -1.7 pp에 그쳐, **복소수 표현이 추론 능력을 효과적으로 보존**함을 시사한다.

**⑤ GPTQ-2bit 대비 우위**:

$$\text{PPL}_{\text{GPTQ-2bit}} = 12.4 \gg \text{PPL}_{\text{FairyFuse}} = 5.52$$

동일 2비트 저장에서 표준 GPTQ-2bit 대비 perplexity가 $6.88$ 포인트 낮다는 것은, 복소수 QAT가 훈련 중 양자화 오차를 구조적으로 보상하여 **일반화 성능을 대폭 향상**시킴을 의미한다.

### 3.2 일반화 성능의 구조적 원천

**광선형 변환의 역할**: 식 (1)의 $\mathbf{y} = \mathbf{U}\mathbf{x} + \mathbf{W}\bar{\mathbf{x}}$에서 켤레 항 $\mathbf{W}\bar{\mathbf{x}}$는 입력의 실수/허수 부분에 대한 **비대칭 처리**를 가능하게 한다. 이는 방향 선택적(direction-selective) 특성 감지와 유사한 효과를 내어, 복잡한 언어 패턴에 대한 일반화를 지원할 수 있다.

**FairyFuse의 추론 경로**: FairyFuse는 Fairy2i가 달성한 일반화 성능을 **무손실(lossless)로 재현**하는 추론 경로를 제공한다. 즉, FairyFuse 자체가 일반화 성능을 직접 향상시키는 것이 아니라, 복소수 QAT의 일반화 이점을 시스템 수준에서 **정확히 보존**한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① 하드웨어-알고리즘 공동 설계의 새 패러다임**

FairyFuse는 양자화 알고리즘과 하드웨어 실행 패러다임이 긴밀히 결합되어야 함을 실증한다. 기존의 "양자화 → 역양자화 → FP16 연산" 패러다임에서 "양자화 표현 → 직접 비산술 연산" 패러다임으로의 전환을 촉진할 것이다.

**② CPU 추론의 재조명**

GPU 중심의 LLM 서빙 생태계에서 CPU가 삼진수 극단 양자화의 **자연스러운 타깃**임을 Roofline 분석으로 구조적으로 증명했다. 이는 엣지 디바이스, 온디바이스 AI, 프라이버시 민감 서버 분야 연구를 가속할 것이다.

**③ 복소수 신경망의 실용화 가능성**

복소수 신경망이 이론적 관심사에 그쳤던 것과 달리, Fairy2i + FairyFuse는 이를 **실제 배포 가능한 시스템**으로 구현했다. 신호처리, 물리 시뮬레이션 등 복소수 도메인 데이터에 대한 LLM 적용 연구를 자극할 것이다.

**④ 비트 연산 기반 추론 가속의 방향성 제시**

LUT 없이 BMI2 `_pext` + AVX-512 마스크 연산만으로 경쟁력 있는 성능을 달성한 것은, 향후 ISA 확장 설계(차세대 Intel/AMD 프로세서의 삼진수 전용 명령어 등) 방향에 영향을 줄 수 있다.

### 4.2 앞으로 연구 시 고려할 점

**① 다양한 모델 및 아키텍처 검증**

현재 LLaMA-2-7B 단일 모델에만 평가되었다. 향후 연구에서는:
- 13B, 70B 등 대형 모델에서의 스레드-행 매핑 최적화
- Mistral, Qwen, LLaMA-3 등 다양한 아키텍처에서의 검증
- MoE(Mixture of Experts) 구조에서의 삼진수 적용 가능성

**② ARM 플랫폼 확장**

ARM NEON은 128비트 SIMD 레인을 사용하며 `_pext`에 해당하는 명령어가 없다. ARM SVE(Scalable Vector Extension) 또는 Apple M-시리즈 NEON에 맞는 비트 디코드 경로 설계가 필요하다. T-MAC이 ARM에서 4× 가속을 보고한 것과의 직접 비교도 중요하다.

**③ 배치 GEMM 및 프리필 최적화**

현재는 단일 토큰 디코딩(GEMV)에 최적화되어 있다. 프롬프트 처리 단계(prefill)는 GEMM이 필요하며, 삼진수 패킹의 이점이 줄어들 수 있다. 투기적 디코딩(Speculative Decoding)과의 결합도 연구 과제다.

**④ 다양한 삼진수 레시피와의 호환성**

현재 Fairy2i에만 의존한다. BitNet b1.58, PB-LLM, OneBit 등 다른 삼진수/이진수 양자화 방법과의 호환 레이어 설계가 필요하며, 특히 **실수 도메인 삼진수 모델**에도 FairyFuse의 마스크 연산 원리를 적용할 수 있는지 연구해야 한다.

**⑤ KV 캐시 양자화와의 결합**

현재 가중치만 삼진수화하고 KV 캐시는 FP16을 사용한다. 활성화(Activation)와 KV 캐시의 극단 양자화를 결합하면 메모리 압박이 더욱 완화될 수 있으나, 복소수 표현과의 상호작용을 주의 깊게 분석해야 한다.

**⑥ 정확도-속도 트레이드오프의 세밀한 분석**

WinoGrande에서 -2.3 pp 하락이 가장 크게 관찰되었다. 이는 상식 추론 태스크에서 삼진수 표현의 한계를 시사할 수 있으며, **혼합 정밀도(Mixed Precision)** 전략(중요 레이어 선택적 고정밀도 유지)과의 결합 연구가 필요하다.

**⑦ 에너지 효율 및 열 특성 분석**

곱셈을 덧셈/뺄셈으로 대체함으로써 전력 소비가 어떻게 변화하는지에 대한 체계적 분석이 아직 없다. 배터리 기반 엣지 디바이스 배포를 위해서는 에너지 효율(tok/J) 메트릭이 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 LLM 양자화 알고리즘 비교

| 연구 | 연도 | 비트 수 | 방법 | PPL (LLaMA-2-7B) | 특징 |
|------|------|---------|------|------------------|------|
| **GPTQ** [Frantar et al., ICLR 2023] | 2023 | 4-bit | 레이어별 2차 최적화 | ~5.65 | 빠른 PTQ, GPU 특화 |
| **AWQ** [Lin et al., MLSys 2024] | 2024 | 4-bit | 활성화 인식 가중치 양자화 | ~5.60 | 중요 채널 보호 |
| **SmoothQuant** [Xiao et al., ICML 2023] | 2023 | W8A8 | 활성화 이상치 스무딩 | - | 가중치+활성화 동시 양자화 |
| **QuIP#** [Tseng et al., ICML 2024] | 2024 | 2-bit | 비정합 처리 + 격자 코드북 | ~5.8 | 2비트 고품질 |
| **AQLM** [Egiazarian et al., ICML 2024] | 2024 | 2-3bit | 가산 양자화 | ~5.9 | 2-3비트 경쟁력 |
| **BitNet b1.58** [Ma et al., 2024] | 2024 | 1.58-bit | 삼진수 스크래치 훈련 | - | 처음부터 훈련 |
| **Fairy2i** [Wang et al., 2025] | 2025 | 2-bit | 복소수 QAT | ~5.52 | PTQ 기반, 복소수 표현 |
| **GPTQ-2bit** [비교용] | 2023 | 2-bit | 2차 최적화 2비트 | 12.4 | 품질 급락 |

### 5.2 CPU 추론 시스템 비교

| 시스템 | 방법 | 비트 | 플랫폼 | 곱셈 | LUT | 처리량 |
|--------|------|------|--------|------|-----|--------|
| **llama.cpp Q4_K_M** | 역양자화 + FMA | 4.2 | x86/ARM | O(nm) | 0 | 26.15 tok/s |
| **llama.cpp Q2_K** | 역양자화 + FMA | 2.5 | x86/ARM | O(nm) | 0 | 20.10 tok/s |
| **T-MAC** [Wei et al., EuroSys 2025] | LUT (VPSHUFB) | 1-2 | ARM M2 | 0 | O(nm) | 4× vs llama.cpp |
| **BitNet.cpp** [Microsoft, 2024] | LUT 기반 | 1.58 | x86/ARM | 0 | O(nm) | - |
| **FairyFuse (ours)** | 마스크 Add/Sub | 2.0 | x86 AVX-512 | **0** | **0** | **32.43 tok/s** |

**핵심 차별점**:

$$\text{연산 복잡도}: \underbrace{\text{llama.cpp}}_{\text{O}(nm) \text{ MUL}} > \underbrace{\text{T-MAC/BitNet.cpp}}_{\text{O}(nm) \text{ LUT}} > \underbrace{\text{FairyFuse}}_{\text{0 MUL, 0 LUT}}$$

### 5.3 효율적 LLM 서빙 시스템 비교

| 시스템 | 주요 기여 | 타깃 | FairyFuse와의 관계 |
|--------|----------|------|------------------|
| **FlashAttention** [Dao et al., NeurIPS 2022] | IO-인식 어텐션 타일링 | GPU | 상호 보완적 (어텐션 vs. 선형 레이어) |
| **vLLM** [Kwon et al., SOSP 2023] | PagedAttention KV 캐시 | GPU | 상호 보완적 (메모리 관리) |
| **DeepSpeed-Inference** [Aminabadi et al., SC 2022] | 다중 GPU 병렬 추론 | GPU 클러스터 | 다른 배포 환경 |
| **QServe** [Lin et al., MLSys 2025] | W4A8KV4 시스템 공동 설계 | GPU | GPU 특화 vs. CPU 특화 |

---

## 참고 자료 (출처)

본 분석은 다음 논문 및 자료를 직접 참조하였습니다:

1. **Zuo, F., Xi, X., Zeng, Q., Wang, F., & Leung, H. F. (2026).** *FairyFuse: Multiplication-Free LLM Inference on CPUs via Fused Ternary Kernels.* arXiv:2604.20913v1 [cs.LG]. *(본 논문)*

2. **Wang, F., et al. (2025).** *Fairy2i: Training complex LLMs from real LLMs with all parameters in {±1, ±i}.* arXiv:2512.02901.

3. **Ma, S., et al. (2024).** *The era of 1-bit LLMs: All large language models are in 1.58 bits.* arXiv:2402.17764.

4. **Wei, J., et al. (2025).** *T-MAC: CPU renaissance via table lookup for low-bit LLM deployment on edge.* EuroSys 2025.

5. **Frantar, E., et al. (2023).** *GPTQ: Accurate post-training quantization for generative pre-trained transformers.* ICLR 2023.

6. **Lin, J., et al. (2024).** *AWQ: Activation-aware weight quantization for LLM compression and acceleration.* MLSys 2024.

7. **Xiao, G., et al. (2023).** *SmoothQuant: Accurate and efficient post-training quantization for large language models.* ICML 2023.

8. **Tseng, A., et al. (2024).** *QuIP#: Even better LLM quantization with Hadamard incoherence and lattice codebooks.* ICML 2024.

9. **Egiazarian, V., et al. (2024).** *Extreme compression of large language models via additive quantization.* ICML 2024.

10. **Dao, T., et al. (2022).** *FlashAttention: Fast and memory-efficient exact attention with IO-awareness.* NeurIPS 2022.

11. **Kwon, W., et al. (2023).** *Efficient memory management for large language model serving with PagedAttention.* SOSP 2023.

12. **Williams, S., Waterman, A., & Patterson, D. (2009).** *Roofline: An insightful visual performance model for multicore architectures.* Communications of the ACM, 52(4):65–76.

13. **Microsoft. (2024).** *BitNet.cpp: Official inference framework for 1-bit LLMs.* https://github.com/microsoft/BitNet

14. **llama.cpp contributors. (2024).** *llama.cpp: Inference of Meta's LLaMA model in C/C++.* https://github.com/ggerganov/llama.cpp

15. **Intel Corporation. (2024).** *Intel Intrinsics Guide.* https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

16. **Gao, L., et al. (2024).** *A framework for few-shot language model evaluation (LM Evaluation Harness).* https://github.com/EleutherAI/lm-evaluation-harness
