# End-to-End Test-Time Training for Long Context (TTT-E2E)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **롱 컨텍스트 언어 모델링(long-context language modeling)을 아키텍처 설계 문제가 아닌 연속 학습(continual learning) 문제**로 재정의합니다. 핵심 통찰은 다음과 같습니다:

> "테스트 시간에 모델이 주어진 컨텍스트를 읽으면서 next-token prediction을 통해 계속 학습하고, 그 컨텍스트를 가중치(weights)로 압축하면 어떨까?"

### 주요 기여

| 기여 | 설명 |
|---|---|
| **E2E 공식화** | 테스트 시간(next-token prediction)과 훈련 시간(meta-learning) 모두에서 End-to-End 최적화 |
| **컨텍스트 스케일링** | 3B 모델 기준 128K 컨텍스트까지 Full Attention과 동일한 스케일링 달성 |
| **추론 효율성** | RNN처럼 상수 추론 지연(constant inference latency) → 128K에서 Full Attention 대비 **2.7× 빠름** |
| **표준 인프라 활용** | 커스텀 커널 없이 표준 Transformer 인프라 사용 가능 |
| **새로운 관점 제시** | 롱 컨텍스트 TTT가 KVB(Key-Value Binding) 기반 메모리제이션이 아닌 일반화(generalization)임을 증명 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

**Full Attention의 딜레마:**

$$\text{Full Attention 복잡도: } O(T^2) \text{ for prefill, } O(T) \text{ for decode}$$

컨텍스트 길이 $T$가 커질수록 계산 비용이 기하급수적으로 증가합니다. 반면 RNN(Mamba 2, Gated DeltaNet 등)은 $O(1)$ 디코드 비용이지만, 긴 컨텍스트에서 성능이 저하됩니다.

**문제 정리:**
- Full Attention: 긴 컨텍스트 활용 우수 but 높은 계산 비용
- RNN/SWA: 낮은 계산 비용 but 긴 컨텍스트에서 성능 저하
- **TTT-E2E 목표**: Full Attention 수준의 컨텍스트 활용 + RNN 수준의 계산 효율성

---

### 2.2 제안 방법 (수식 포함)

#### Step 1: 기본 TTT via Next-Token Prediction (Subsection 2.1)

모델 $f$, 가중치 $W$로 정의되는 표준 next-token prediction loss:

$$\ell_t(W) = \text{CE}(f(x_{t-1}; W),\, x_t) $$

테스트 시간에 순차적으로 gradient descent:

$$W_t = W_{t-1} - \eta \nabla \ell_t(W_{t-1}) $$

최종 예측: $\hat{p}_{T+1} = f(x_T; W_T)$

- **복잡도**: Prefill $O(T)$, Decode $O(1)$

---

#### Step 2: E2E 메타 학습 (Subsection 2.2)

테스트 손실(실제 평가 손실)은 TTT를 수행한 가중치에 대한 손실:

$$\mathcal{L}(W_0; X) = \frac{1}{T} \sum_{t=1}^{T} \ell_t(W_{t-1}) = \frac{1}{T} \sum_{t=1}^{T} \text{CE}(f(x_{t-1}; W_{t-1}),\, x_t) $$

**TTT-naive (비교 대상)**: 업데이트를 고려하지 않고 정적 가중치로 훈련:

$$\mathcal{L}_{\text{naive}}(W_0; X) = \frac{1}{T} \sum_{t=1}^{T} \ell_t(W_0) $$

**핵심 차이**: $\mathcal{L}$은 TTT 후의 손실을 직접 최적화(E2E)하지만, $\mathcal{L}_{\text{naive}}$는 정적 모델의 손실만 최적화합니다. $\mathcal{L}$을 최소화하려면 $\nabla \mathcal{L}(W_0)$를 계산해야 하며, 이는 **gradients of gradients** (메타-그래디언트)를 필요로 합니다. 이것이 MAML 스타일의 메타러닝에 해당합니다.

---

#### Step 3: Mini-Batch TTT + Sliding Window (Subsection 2.3)

배치 크기 $b$로 일반화된 내부 루프(inner loop) 업데이트:

$$W_i = W_{i-1} - \eta \cdot \frac{1}{b} \sum_{t=(i-1)b+1}^{ib} \nabla \ell_t(W_{i-1}), \quad i = 1, \ldots, T/b $$

대응하는 훈련 손실(외부 루프, outer loop):

$$\mathcal{L}(W_0; X) = \frac{1}{T} \sum_{i=1}^{T/b} \sum_{t=(i-1)b+1}^{ib} \ell_t(W_{i-1}) $$

- $b=1$이면 식 (2), (3)으로 환원됩니다.
- Mini-batch로 인한 미니배치 내부 컨텍스트 손실 문제를 **Sliding-Window Attention (SWA)**으로 보완합니다.

**하이퍼파라미터 설정 (주요 실험):**
- 슬라이딩 윈도우 크기: $k = 8\text{K}$
- TTT 미니배치 크기: $b = 1\text{K}$
- 조건: $k \geq b$ (미니배치 내 컨텍스트를 SWA로 기억하기 위해)

---

#### 대안 유도: TTT-KVB → TTT-E2E (Subsection 2.4)

TTT-KVB(Key-Value Binding)의 층별 재구성 손실:

$$\ell_t^{(l)}\!\left(W_{t-1}^{(l)}\right) = \left\|g\!\left(\theta_K^{(l)} x_t^{(l)};\, W_{t-1}^{(l)}\right) - \theta_V^{(l)} x_t^{(l)}\right\|^2 $$

$$z_t^{(l)} = g\!\left(\theta_Q^{(l)} x_t^{(l)};\, W_t^{(l)}\right) $$

TTT-E2E는 이 층별 손실을 **단일 next-token prediction 손실**로 대체하고, 출력 규칙을 단순화:

$$z_t^{(l)} = g\!\left(\theta_K^{(l)} x_t^{(l)};\, W_{t-1}^{(l)}\right) $$

→ $\theta_K$, $\theta_V$, $\theta_Q$ 분리가 불필요해지고, 전체 네트워크 끝에서 단일 CE 손실로 TTT를 수행합니다.

---

### 2.3 모델 구조

```
[전체 구조: L개 블록]
├── 하위 3L/4 블록 (고정, TTT 미적용)
│   ├── Sliding-Window Attention (k=8K)
│   └── MLP (고정)
│
└── 상위 L/4 블록 (TTT 적용)
    ├── Sliding-Window Attention (k=8K, 고정)
    ├── MLP (Static, 사전 지식 보존용)   ← 추가된 두 번째 MLP
    └── MLP (Dynamic, TTT로 업데이트됨) ← 핵심 "hidden state"
```

**구현 세부사항:**

| 설계 선택 | 이유 |
|---|---|
| MLP 레이어만 TTT | 임베딩/정규화/어텐션 레이어 업데이트 시 outer loop 불안정 |
| 마지막 1/4 블록만 TTT | 계산 비용 vs. 컨텍스트 압축 용량의 트레이드오프 |
| 블록당 두 개의 MLP | 사전학습 지식 망각 방지 (static MLP로 보존) |

**상태 크기 비교** (760M 모델 기준):
- TTT-E2E: hidden state 88M 파라미터
- TTT-KVB: hidden state 18M 파라미터 (LoRA 사용)
- → TTT-E2E가 **5× 더 큰 hidden state**

---

### 2.4 성능 향상

#### 컨텍스트 길이 스케일링 (핵심 결과)

3B 모델, 164B 토큰 학습 기준:

| 방법 | 128K 컨텍스트 스케일링 | 추론 지연 |
|---|---|---|
| Full Attention | ✅ 완벽한 스케일링 (기준선) | $O(T)$, 가장 느림 |
| SWA | ❌ 악화 | $O(1)$ |
| Mamba 2 | ❌ 악화 | $O(1)$ |
| Gated DeltaNet | ❌ 악화 | $O(1)$ |
| TTT-KVB | ❌ 악화 | $O(1)$ |
| **TTT-E2E** | **✅ Full Attention과 동일한 스케일링** | **$O(1)$, Full Attention 대비 2.7× 빠름** |

#### 훈련 컴퓨팅 스케일링

- 모델 크기 경계: ~760M 이상에서 Full Attention과 유사한 트렌드
- 토큰 수 경계: ~48B 이상에서 Full Attention과 유사한 트렌드
- 더 좋은 토크나이저(Llama 3) 및 데이터(DCLM 2024) 사용 시 추가 개선

#### 토큰 수준 손실 분석

- TTT-E2E는 **전체 컨텍스트 길이에 걸쳐 Full Attention보다 일관되게 낮은 손실** 달성
- 우위는 주로 초반 토큰에서 발생 (메타-학습된 초기화의 집중된 예측 덕분)

#### S-NIAH (Needle in a Haystack) 성능

| 방법 | 8K | 128K (S-NIAH-1) |
|---|---|---|
| Full Attention | 1.00 | **0.99** |
| TTT-E2E | 1.00 | 0.06 |
| Gated DeltaNet | 1.00 | 0.07 |

→ **정밀 회상(lossless recall)** 작업에서는 Full Attention이 압도적으로 우수합니다.

---

### 2.5 한계

1. **훈련 지연(Training Latency)**: 8K 컨텍스트에서 Full Attention 대비 **3.4× 느림** (gradients of gradients 계산 비용). 현재 cuDNN FlashAttention이 이중 그래디언트를 지원하지 않음.

2. **정밀 회상 성능 부재**: S-NIAH에서 압도적으로 열세. 압축 기반 메커니즘의 본질적 한계.

3. **메모리 사용**: TTT의 hidden state로 $W_1, \ldots, W_T$를 저장해야 하므로, gradient checkpointing을 시간 축으로 적용해야 함 (log(T) 계수로 지연 증가).

4. **안정성 제약**: 미니배치 크기 $b < 1\text{K}$는 하드웨어 활용도 저하 및 불안정성 초래.

5. **평가 범위 제한**: 128K까지만 실험, 더 긴 컨텍스트(1M+) 검증 부재.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 TTT-E2E가 일반화를 향상시키는 핵심 메커니즘

#### (A) 집중된 예측 (Focused Prediction)

논문의 핵심 직관:

> "Full Attention의 가중치는 미래의 모든 토큰에서 잘 동작해야 한다. 이는 매우 어려운 과제인데, 가능한 모든 미래에 잘 적응하려면 특정 미래에 잘 적응하는 능력이 제한되기 때문이다. 하지만 TTT-E2E의 가중치는 현재 미니배치의 토큰에만 집중하면 된다."

이는 **작업별 적응(task-specific adaptation)**의 일반화 원리와 일맥상통합니다. 모델이 현재 컨텍스트에 특화된 표현을 학습하므로, 특정 문서/도메인의 분포에 더 잘 맞게 됩니다.

#### (B) E2E 훈련-테스트 정렬

$$\underbrace{\mathcal{L}(W_0; X) = \frac{1}{T}\sum_{t=1}^{T}\ell_t(W_{t-1})}_{\text{훈련 손실}} = \underbrace{\frac{1}{T}\sum_{t=1}^{T}\text{CE}(f(x_{t-1}; W_{t-1}), x_t)}_{\text{테스트 손실}}$$

훈련 손실과 테스트 손실이 **동일**합니다. 이는 일반화의 핵심 조건 중 하나인 훈련-테스트 분포 정렬을 만족합니다. TTT-naive는 이 정렬이 깨져 있어 일반화 보장이 어렵습니다.

#### (C) 메타-학습의 일반화 효과

MAML과 유사하게, 외부 루프가 다양한 훈련 시퀀스에 걸쳐 $W_0$를 최적화하므로, **TTT에 유리한 초기화**를 학습합니다. 이 초기화는 특정 데이터에 과적합되지 않고 다양한 컨텍스트에 빠르게 적응 가능한 일반적 구조를 학습합니다.

#### (D) Full Attention 위에서도 직교적 개선

실험에서 TTT-E2E가 SWA($k=8\text{K}$) = Full Attention 상황에서도 손실을 **0.018 추가 감소**시킵니다. 이는 TTT-E2E가 어텐션 메커니즘의 한계를 보완하는 것이 아니라, **독립적이고 보완적인 일반화 메커니즘**을 제공함을 시사합니다.

### 3.2 일반화 관련 실험적 증거

**Figure 6의 관찰:**
- TTT-E2E는 32K와 128K 양쪽에서 모두 Full Attention보다 낮은 손실 달성
- 컨텍스트 끝부분에서의 차이가 작지만, 컨텍스트 확장 시 교차하지 않음 → **분포 이동에 강인한 일반화**

**Figure 5 (훈련 컴퓨팅 스케일링):**
- 충분한 훈련 예산에서 Full Attention과 동일한 스케일링 트렌드
- 적은 훈련 예산에서는 Full Attention 대비 우위 (이는 RNN 일반적 특성)

### 3.3 분포 이동(Distribution Shift) 시나리오에서의 가능성

논문의 Section 4.2.2는 일반화로서의 TTT를 별도로 논의합니다:

- 테스트 인스턴스가 out-of-distribution일 때, 정적 모델은 적응하기 어렵습니다.
- TTT는 테스트 시점에 실제 데이터를 활용하여 적응하므로, **도메인 이동(domain shift) 상황에서 자연스러운 적응 메커니즘**을 제공합니다.
- 논문이 제시하는 흥미로운 미래 방향: **자기 생성 토큰(self-generated tokens)** 으로 TTT 수행 → RNN의 게이팅 메커니즘처럼 spurious input을 필터링

### 3.4 데이터 품질과 일반화의 관계

논문이 발견한 흥미로운 관찰:
- 더 좋은 토크나이저 (Llama 2 → Llama 3): Full Attention 대비 우위 **+0.01 개선**
- 더 좋은 데이터 (SlimPajama → DCLM 2024): 훈련 토큰 스케일링에서 Full Attention과 동일한 트렌드 달성

→ **데이터 품질이 TTT의 일반화 성능에 critical한 역할**을 함을 시사

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 비교 대상 연구 체계

```
롱 컨텍스트 처리 접근법
├── 아키텍처 설계
│   ├── Sparse/Sliding Attention: Longformer (2020), Native Sparse Attention (2025)
│   └── Hybrid: Gemma 2 (2024), Nemotron-H (2025), Kimi Linear (2025)
├── RNN 기반
│   ├── Mamba 2 (2024)
│   ├── Gated DeltaNet (2024)
│   └── Linear Attention 계열
└── TTT 기반
    ├── Learning to (Learn at Test Time) - Sun et al. (2023, 2024)
    ├── TTT-KVB / TTT-done-right (Zhang et al., 2025)
    ├── Titans (Behrouz et al., 2024)
    ├── MesaNet (von Oswald et al., 2025)
    └── TTT-E2E (본 논문, 2024/2025)
```

### 4.2 주요 방법론 비교 테이블

| 방법 | 출처 | 컨텍스트 스케일링 | 추론 복잡도 | 훈련 복잡도 | E2E 여부 |
|---|---|---|---|---|---|
| **Transformer (Full Attention)** | Vaswani et al., 2017 | ✅ 최고 | $O(T)$ | $O(T^2)$ | - |
| **Longformer** | Beltagy et al., 2020 | 제한적 | $O(T)$ | $O(T)$ | - |
| **Mamba 2** | Dao & Gu, 2024 | ❌ | $O(1)$ | $O(T)$ | - |
| **Gated DeltaNet** | Yang et al., 2024 | ❌ | $O(1)$ | $O(T)$ | - |
| **TTT-KVB (Sun et al.)** | Sun et al., 2024 | 부분적 | $O(1)$ | $O(T)$ | 훈련만 |
| **Titans** | Behrouz et al., 2024 | 부분적 | $O(1)$ | $O(T)$ | 부분 |
| **TTT-done-right (Zhang et al.)** | Zhang et al., 2025 | 부분적 | $O(1)$ | $O(T)$ | 부분 |
| **TTT-E2E (본 논문)** | Tandon et al., 2025 | ✅ Full Attention과 동일 | $O(1)$ | $O(T)$* | **완전 E2E** |

*훈련 지연은 현재 구현에서 병목

### 4.3 세부 방법론 비교

#### TTT-KVB vs. TTT-E2E

| 항목 | TTT-KVB | TTT-E2E |
|---|---|---|
| 손실 함수 | 층별 KV 재구성 손실 (식 7) | 단일 next-token prediction (식 3) |
| 추가 파라미터 | $\theta_K, \theta_V, \theta_Q$ (각 블록) | 없음 |
| 업데이트 레이어 | 모든 블록 | 마지막 1/4 블록 |
| Hidden state 크기 | 18M (760M 모델) | 88M (760M 모델) |
| E2E at test time | ❌ | ✅ |
| E2E at train time | ✅ | ✅ |
| 추론 지연 | 0.017 sec/1K tokens | 0.0086 sec/1K tokens |

**핵심 발견**: 층별 재구성 손실 → next-token prediction 손실로 교체만으로 성능이 TTT-E2E 수준으로 향상됨 (Table 1).

#### Mamba 2 / Gated DeltaNet vs. TTT-E2E

- **공통점**: 모두 $O(1)$ 추론, 일종의 RNN으로 해석 가능
- **차이점**: Mamba 2와 Gated DeltaNet은 **선형 hidden state** + 커스텀 커널 필요
  - TTT-E2E는 비선형 MLP를 hidden state로 사용 → **더 큰 표현력**
  - TTT-E2E는 커스텀 커널 불필요 → 표준 인프라로 더 큰 state 허용

TTT-E2E의 관점에서 Gated DeltaNet의 delta rule은 TTT-KVB의 선형 모델 특수 케이스로 해석 가능합니다 (논문 각주 1).

#### Titans vs. TTT-E2E

- Titans도 TTT 기반이며 "learning to memorize at test time"을 표방
- 하지만 TTT-KVB 구조를 따르므로 층별 재구성 손실 사용 → E2E at test time 아님
- TTT-E2E는 단일 글로벌 손실로 전체 네트워크를 최적화 → 더 큰 hidden state 효율적 활용

#### Clark et al. (2022) "Meta-learning Fast Weight Language Models" vs. TTT-E2E

- 가장 유사한 방법론: Full Attention + MLP fast weights, next-token prediction으로 업데이트
- **차이**: Fast weights를 네트워크 끝에만 추가 (interleaving 없음)
- 실험에서 interleaving이 성능 유지에 critical하다는 것이 TTT-E2E에서 확인됨
- 선형 복잡도 달성 못함 (full attention 유지)

### 4.4 AlphaProof, ARC-AGI 맥락에서의 TTT

| 응용 | 방법 | TTT 유형 |
|---|---|---|
| AlphaProof (IMO 2024) | 테스트 문제별 강화학습 | TTT for Novel Instances |
| ARC-AGI (Akyurek et al., 2024) | Few-shot 예시 증강 후 SL | TTT on Nearest Neighbors |
| **TTT-E2E** | 시퀀스 압축 | TTT on Sequences |

→ TTT-E2E는 이 중 **시퀀스 상의 TTT** 범주로, 언어 모델링의 순차적 특성에 특화된 접근법입니다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5.1 향후 연구에 미치는 영향

#### (A) 패러다임 전환: 아키텍처 → 학습 알고리즘

TTT-E2E는 롱 컨텍스트 문제를 새로운 아키텍처 발명이 아닌 **학습 알고리즘 설계 문제**로 재정의합니다. 이는:
- 기존 Transformer 아키텍처를 재활용하면서 효율성 달성 가능
- 아키텍처 탐색 비용 절감 가능성
- 다른 modality (비디오, 로봇 등)에도 동일한 프레임워크 적용 가능성

#### (B) 연속 학습(Continual Learning)과 LLM의 결합

TTT-E2E는 사실상 **시퀀스 수준의 연속 학습**을 수행합니다. 이는:
- 동적으로 변하는 지식 기반에 적응하는 LLM 연구로 이어질 수 있음
- 개인화(personalization): 각 사용자 세션에 특화된 적응
- 도메인 적응: 특정 전문 분야 문서에 즉각 적응

#### (C) 메타-학습과 LLM의 결합 심화

MAML과 유사한 이중 루프 최적화가 실제 대규모 언어 모델에서 작동함을 입증합니다:
- 더 발전된 메타 최적화 알고리즘(예: Meta-SGD, MAML++)의 LLM 적용 연구
- 외부 루프(outer loop) 최적화의 더 효율적인 구현 연구

#### (D) 추론 시간 적응(Inference-Time Adaptation)의 새로운 방향

Test-Time Compute 스케일링 트렌드(o1, DeepSeek-R1 등)와 맥을 같이 하여:
- 추론 시간에 더 많은 계산을 투입할수록 더 좋은 성능
- TTT-E2E는 "더 많은 컨텍스트 = 더 많은 적응 = 더 좋은 성능"의 새로운 스케일링 축 제시

#### (E) 생물학적 메모리 계층 구조 영감

논문 결론에서 제시하듯:

$$\text{TTT-E2E} = \underbrace{\text{SWA (단기 기억)}}_{\text{작업 기억}} + \underbrace{\text{업데이트된 MLP (장기 기억)}}_{\text{맥락 압축}}$$

이 계층적 메모리 구조가 신경과학 영감 AI 연구에 새로운 방향을 제시합니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (A) 훈련 효율성 개선 (가장 긴급한 과제)

1. **cuDNN FlashAttention 호환 커스텀 커널 개발**
   - 현재 cuDNN FlashAttention은 이중 그래디언트 미지원
   - 커스텀 어텐션 커널이 하드웨어 활용도를 크게 개선할 수 있음

2. **사전학습된 Transformer에서 초기화**
   - 논문이 제안하는 방향: 기존 Transformer를 TTT-E2E의 시작점으로 활용
   - RNN 계열 연구(Mamba in the Llama 등)에서 검증된 기법
   - TTT 훈련 비용을 전체 학습 비용의 소수 비율로 제한 가능

3. **그래디언트 체크포인팅 최적화**
   - 현재 시간 축의 log(T) 계수가 지연 증가를 유발
   - 더 효율적인 메모리-계산 트레이드오프 탐색

#### (B) 정밀 회상 성능 개선

S-NIAH에서 Full Attention 대비 큰 차이 → 압축 기반 방법의 근본적 한계

**가능한 방향:**
- TTT-E2E + 선택적 Key-Value 캐싱 하이브리드
- 중요도 기반 컨텍스트 필터링으로 압축 손실 최소화
- 외부 검색 메커니즘(RAG)과의 결합

#### (C) 자기 생성 토큰을 활용한 TTT

논문이 제안한 미래 방향 중 가장 흥미로운 것:

$$\text{TTT-E2E with Self-Generation} = \text{현재 미니배치} + \text{자기 생성 보완 데이터}$$

- 현재 미니배치의 필터링/재구성 버전으로 TTT 수행
- 이전 미니배치 복습 (리뷰 메커니즘)
- RNN 게이팅처럼 spurious input 방어 효과

#### (D) 더 긴 컨텍스트 (1M+) 검증

- 현재 실험은 최대 128K
- 실제 응용(법률 문서, 코드베이스, 긴 소설 등)은 더 긴 컨텍스트 요구
- 블록 수의 1/4 업데이트 규칙이 더 긴 컨텍스트에서도 유효한지 검증 필요

#### (E) 명령어 튜닝 및 RLHF와의 통합

- 현재 논문은 base model에 초점
- 명령어 튜닝 후 TTT-E2E의 행동 변화 연구 필요
- 강화학습(RL) 훈련 중 TTT-E2E의 역할 (예: long chain-of-thought 생성 지원)

#### (F) 다중 모달리티 적용

TTT의 순차적 적응 특성은 다른 순차적 데이터에도 자연스럽게 확장:
- **비디오**: 이미 Sun et al.의 이전 연구에서 검증 (video streams TTT)
- **로봇 공학**: 정책 적응 (Hansen et al., 2020)
- **멀티모달 LLM**: 비디오-텍스트 혼합 긴 컨텍스트

#### (G) 이론적 분석의 필요성

현재 논문은 실험적 검증에 집중하며, 다음에 대한 이론적 분석이 부족합니다:
- TTT의 일반화 오차 경계 (generalization error bound)
- 어떤 조건에서 TTT가 static 모델보다 유리한가?
- 최적 배치 크기 $b$와 윈도우 크기 $k$의 이론적 관계

#### (H) 장기 디코딩(Long Decoding) 시나리오

- 현재 디코딩 평가는 8K 추가 토큰에 한정
- 더 긴 추론(긴 사고 사슬) 시나리오에서 TTT self-training의 효과 검증 필요
- 디코딩 중 TTT 업데이트의 최적 타이밍 전략

---

## 참고자료

- **주 논문**: Tandon, A., Dalal, K., Li, X., et al. "End-to-End Test-Time Training for Long Context." arXiv:2512.23675v2 (31 Dec 2025). [제공된 PDF]

- **관련 논문 (논문 내 인용)**:
  - Sun et al. "Learning to (learn at test time): RNNs with expressive hidden states." arXiv:2407.04620, 2024.
  - Sun et al. "Learning to (learn at test time)." arXiv:2310.13807, 2023.
  - Zhang et al. "Test-time training done right." arXiv:2505.23884, 2025.
  - Dao & Gu. "Transformers are SSMs: Mamba 2." arXiv:2405.21060, 2024.
  - Yang et al. "Gated Delta Networks." arXiv:2412.06464, 2024.
  - Behrouz et al. "Titans: Learning to memorize at test time." arXiv:2501.00663, 2024.
  - Finn et al. "Model-agnostic meta-learning (MAML)." ICML, 2017.
  - Clark et al. "Meta-learning fast weight language models." arXiv:2212.02475, 2022.
  - Beltagy et al. "Longformer." arXiv:2004.05150, 2020.
  - Schmidhuber. "Learning to control fast-weight memories." Neural Computation, 1992.
  - Akyürek et al. "The surprising effectiveness of test-time training for abstract reasoning." arXiv:2411, 2024.
  - Gu & Dao. "Mamba." arXiv:2312.00752, 2023.
  - Hoffmann et al. "Training compute-optimal large language models (Chinchilla)." 2022.
  - Hubert et al. "AlphaProof." Nature, 2025.
  - von Oswald et al. "MesaNet." arXiv:2506.05233, 2025.
