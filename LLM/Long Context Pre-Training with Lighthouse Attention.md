# Long Context Pre-Training with Lighthouse Attention

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

"Long Context Pre-Training with Lighthouse Attention" (Peng et al., arXiv:2605.06554v1, 2026)의 핵심 주장은 다음과 같습니다:

> **훈련 시 Lighthouse Attention을 사용한 후 짧은 Dense-SDPA 재개(Resumption) 단계를 거치면, 동일 토큰 예산으로 처음부터 Dense SDPA로 훈련한 모델과 동등하거나 그보다 낮은 손실(loss)을 달성할 수 있다.**

즉, 계층적(hierarchical) 희소(sparse) 훈련이 추론(inference) 시 full attention 능력을 손상시키지 않음을 실험적으로 입증합니다.

### 세 가지 주요 기여

| 기여 | 설명 |
|------|------|
| **(i) 대칭적 계층 압축** | $Q, K, V$를 동시에 평균 풀링(average pooling)하여 다중 해상도 피라미드 구성 |
| **(ii) 파라미터-프리 선택** | $\ell_2$ 노름 기반의 gradient-free top-K 선택으로 복잡한 역전파 커널 불필요 |
| **(iii) 2단계 훈련 레시피** | Stage 1: Lighthouse로 대부분 훈련 → Stage 2: Dense SDPA로 짧게 복원 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 이차 복잡도(Quadratic Complexity)**

표준 Scaled Dot-Product Attention(SDPA)의 시간 및 메모리 복잡도는 다음과 같습니다:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right)V, \quad \text{비용: } \Theta(N^2 d)$$

FlashAttention은 상수를 줄이지만 점근적 복잡도( $\Theta(N^2)$ )는 제거하지 못합니다. $N \geq 10^5$에서 이 항이 지배적입니다.

**문제 2: 기존 희소 어텐션의 두 가지 설계 결함**

- **(i) 비대칭성(Asymmetry):** 기존 방법(NSA, HISA, InfLLM-V2 등)은 쿼리($Q$)는 full resolution을 유지하고 키·값($K, V$)만 풀링 → 멀티스케일 표현이 아닌 압축 메모리에 불과
- **(ii) 구조적 결합(Architectural Entanglement):** 선택 로직이 어텐션 커널 내부에 내장 → 최적화된 Dense 커널(FlashAttention) 재사용 불가

**문제 3: 훈련-시 정확성(Training-time Correctness)**

추론 전용 희소 방법은 Dense 백본의 정확도를 상속받습니다. 그러나 **훈련-시 희소 방법은 더 어려운 질문을 직면합니다: 훈련 완료 후 모델이 여전히 유능한 Dense-attention 모델인가?**

---

### 2.2 제안하는 방법: Lighthouse Attention

#### 전체 파이프라인 (4단계)

```
X → [Projections] → Q,K,V → [Pyramid Pool] → {Q^(ℓ),K^(ℓ),V^(ℓ)}
                                    ↓
                          [Hierarchical Selector]
                          Score → Top-K → I
                                    ↓
                          [Dense Gather] → Q̃,K̃,Ṽ
                                    ↓
                          [Stock FlashAttention] → Õ
                                    ↓
                          [Scatter-back] → O
```

#### 단계 (i): 피라미드 구성 (Pyramid Construction)

$Q, K, V \in \mathbb{R}^{N \times d}$에 대해 $L$-레벨 피라미드를 구성합니다. 레벨 $\ell$에서 $i$번째 윈도우:

$$\mathcal{W}_i^{(\ell)} = \left[i p^\ell,\ (i+1)p^\ell - 1\right], \quad i = 0, \ldots, \frac{N}{p^\ell} - 1 $$

피라미드 엔트리는 평균 풀링으로 생성:

```math
Q_i^{(\ell)} = \text{Pool}_\mu\!\left\{Q_j \mid j \in \mathcal{W}_i^{(\ell)}\right\}, \quad K_i^{(\ell)} = \text{Pool}_\mu\!\left\{K_j \mid j \in \mathcal{W}_i^{(\ell)}\right\}, \quad V_i^{(\ell)} = \text{Pool}_\mu\!\left\{V_j \mid j \in \mathcal{W}_i^{(\ell)}\right\}
```

총 피라미드 엔트리 수: $\sum_{\ell=0}^{L-1} N/p^\ell \leq N \cdot p/(p-1)$, 즉 $\Theta(N)$ 시간·메모리.

**핵심 차별점:** 기존 방법(NSA, HISA)과 달리 $Q$도 $K, V$와 동시에(대칭적으로) 풀링합니다. 이로써 $(Q^{(\ell)}, K^{(\ell)}, V^{(\ell)})$이 동일 표현 공간에 존재하는 일관된 트리플을 형성합니다.

#### 단계 (ii): 스코어링 및 선택 (Scoring & Selection)

레벨 0에서 per-head $\ell_2$ 노름으로 스코어 계산:

$$s_{0,i}^{QK} = \|Q_i\|_2, \quad s_{0,i}^{KQ} = \|K_i\|_2, \quad i = 0, \ldots, N-1 $$

상위 레벨에서는 레벨 0에서 max-pool로 상속:

$$s_{\ell,i}^{QK} = \max_{0 \leq j < p^\ell} s_{0,\, ip^\ell + j}^{QK}, \quad s_{\ell,i}^{KQ} = \max_{0 \leq j < p^\ell} s_{0,\, ip^\ell + j}^{KQ} $$

모든 레벨에 걸친 Top-K 선택:

```math
\mathcal{I} = \text{TopK}\!\left(\left\{s_{\ell,i}^{QK},\ s_{\ell,i}^{KQ} : (\ell, i) \in \mathcal{P}\right\},\ k\right)
```

> **설계 선택:** Top-K는 **비미분가능(non-differentiable)**하며 Straight-Through Estimator나 Gumbel-Softmax를 사용하지 않습니다. 그래디언트는 선택 인덱스를 통해서가 아니라 $\tilde{Q}, \tilde{K}, \tilde{V}$를 통해 $W_Q, W_K, W_V$로 흐릅니다.

#### 단계 (iii): 집합된 부분 시퀀스 어텐션 (Gathered-Sequence Attention)

$\mathcal{I}$에서 연속적인 부분 시퀀스 조립:

$$\tilde{Q}_m = Q_{i_m}^{(\ell_m)}, \quad \tilde{K}_m = K_{i_m}^{(\ell_m)}, \quad \tilde{V}_m = V_{i_m}^{(\ell_m)}, \quad (\ell_m, i_m) \in \mathcal{I},\ m = 1,\ldots,S $$

부분 시퀀스 길이:

$$S = \frac{N}{p^{L-1}} + (L-1)\,p\,k $$

예시: $N = 10^6,\ L = 4,\ p = 4,\ k = 4096$이면 $S \approx 6.5 \times 10^4 \ll N$

Stock FlashAttention으로 어텐션 계산:

$$\tilde{O} = \text{Attn}\!\left(\tilde{Q}, \tilde{K}, \tilde{V}; \tilde{M}\right) $$

#### 단계 (iv): Scatter-Back 재구성

레벨 $\ell$, 위치 $i$의 출력을 shifted range에 기록:

$$\mathcal{R}(\ell, i) = \left[i p^\ell + p^\ell - 1,\ i p^\ell + 2p^\ell - 2\right] $$

위치별 최종 출력 (레벨 간 기여 합산):

$$O_j = \sum_{m:\, j \in \mathcal{R}(\ell_m, i_m)} \tilde{O}_m $$

---

### 2.3 점근적 복잡도 분석

$L = \log_p(N/k)$로 설정하면 $p^{L-1} = N/(pk)$, 이를 식 (8)에 대입:

$$S = pk + (L-1)pk = pk \cdot L = pk \cdot \log_p(N/k) = \Theta(k \log N) $$

Dense FlashAttention 비용:

$$S^2 \cdot d = \Theta(k^2 \log^2 N \cdot d) \quad \text{(N에 대해 다항로그)}$$

전체 per-layer 계산 비용:

$$T_\text{layer} = \Theta(T \cdot d) + \Theta(T \log k) + \Theta(k^2 \log^2 T \cdot d) \xrightarrow{\text{bounded }k} \Theta(T \cdot d) $$

**어텐션 방법론 비교:**

| 방법 | Per-layer 계산 복잡도 |
|------|----------------------|
| Dense Softmax | $\Theta(T^2 \cdot d)$ |
| Log-Linear Attention | $\Theta(T \log T \cdot d)$ |
| **Lighthouse (bounded $k$)** | $\mathbf{\Theta(T \cdot d)}$ |
| Linear Attention / SSMs | $\Theta(T \cdot d)$ |

---

### 2.4 모델 구조

**실험 아키텍처:**
- 530M 파라미터 Llama-3 스타일 디코더
- $d_\text{model} = 1024$, 30 레이어, $H = 8$ 헤드, head dim 128
- 레이어 $\{0, 1, 28, 29\}$: Dense SDPA 유지
- 나머지 26개 레이어: Lighthouse Attention 적용

**2단계 훈련 레시피:**

```
Stage 1 (10k~12k steps):  Lighthouse Attention으로 훈련
Stage 2 (4k~6k steps):    Dense SDPA로 재개 (동일 optimizer state 유지)
총 예산: 16,000 steps ≈ 50.3B tokens
```

---

### 2.5 성능 향상

#### SDPA 복원 가능성 (Recoverability)

| 설정 | B200-Hrs ↓ | Tok/s (k) ↑ | Final Loss ↓ |
|------|-----------|-------------|--------------|
| SDPA Baseline (ctx=98k) | 303.2 | 45.6 | 0.7237 |
| LH→SDPA (12k+4k) | 214.7 | 74.7 | 0.7102 |
| LH→SDPA (11k+5k) | 219.6 | 75.4 | 0.7001 |
| LH→SDPA (10k+6k) | 228.0 | 75.0 | **0.6980** |

**모든 Lighthouse 구성이 Dense 베이스라인(0.7237)을 능가합니다.**

#### 속도 향상

- $N = 512K$에서 Forward 패스 **21× 빠름**, Forward+Backward **17.3× 빠름**
- End-to-end 벽시계 속도 **1.40×~1.69× 향상** (동일 토큰 예산)
- Stage-1 처리량: 84~126k tok/s/GPU (Dense SDPA ≈46k 대비 약 2배)

#### 최고 성능 하이퍼파라미터 구성

$$L=3,\ p=2,\ k=1536,\ \text{Dilated Scorer} \rightarrow \text{Final Loss: } 0.6825$$

#### NIAH (Needle-in-a-Haystack) 검색 성능

| 구성 | 평균 검색율 |
|------|-----------|
| $k=2048$, Dilated | **0.76** |
| $k=1536$, Dilated | 0.73 |
| $k=2048$, Norm | 0.72 (베이스라인 동등) |
| Dense SDPA Baseline | 0.72 |
| $k=1536$, Norm | 0.65 |

---

### 2.6 한계점

1. **대칭 풀링과 자기회귀 추론 불일치:** $Q/K/V$ 대칭 풀링은 모든 쿼리가 하나의 forward pass에 공존한다고 가정하지만, 자기회귀 디코딩은 이를 위반합니다. 따라서 추론 가능 체크포인트를 얻으려면 Dense-SDPA 복원 단계가 필수입니다.

2. **선형이 아닌 준이차(Sub-quadratic) 복잡도:** 내부 어텐션은 수집된 부분 시퀀스에 대해 $\Theta(S^2 d)$로, $k$가 $N$과 함께 스케일해야 하는 체제에서는 특성화가 미흡합니다.

3. **소규모 실험:** 530M 파라미터 모델에서의 검증으로, 더 큰 스케일(7B, 70B 등)에서의 동작은 미검증입니다.

4. **다운스트림 평가 부재:** 모든 평가가 훈련 손실과 NIAH에 국한되어 있으며, 실제 벤치마크(MMLU, HELM, LongBench 등)에서의 성능은 검증되지 않았습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 훈련-정확성 기준 (Training-Correctness Criterion)

Lighthouse의 일반화 성능 향상 가능성의 핵심 근거는 **"SDPA 복원 가능성"** 테스트입니다:

> *"훈련이 끝난 후 Dense-SDPA 재개를 통해 Dense-from-scratch 베이스라인의 품질을 복원하는 것을 우리의 핵심 정확성 기준으로 삼는다."*

이 기준이 중요한 이유:
- 추론 전용 희소 방법(HISA, H2O 등)은 Dense 백본에서 정확성을 상속받으므로 이 테스트가 불필요합니다.
- **훈련-시 희소 방법(MoBA, NSA)**은 이 테스트를 통과해야만 진정한 일반화가 가능합니다.

### 3.2 계층적 훈련 신호의 정규화 효과

논문에서 발견된 흥미로운 현상:

> $k$가 감소함에 따라 최종 손실이 단조적으로 감소: $k \in \{1536, 2048, 3072, 4096\}$에 대해 $0.6825 \rightarrow 0.6880 \rightarrow 0.6890 \rightarrow 0.6951$

저자들은 이를 **"계층적 선택이 우리의 토큰 예산에서 정규화 효과를 가질 수 있다"**고 해석합니다. 이는 Lighthouse의 희소 훈련 신호가 단순한 근사(approximation)가 아니라 **유익한 귀납적 편향(inductive bias)** 을 제공할 수 있음을 시사합니다.

### 3.3 대칭적 Q/K/V 풀링의 멀티스케일 표현

비대칭 방법과의 핵심 차이:

$$\text{비대칭: } Q \in \mathbb{R}^{N \times d},\ K^{(\ell)} \in \mathbb{R}^{N/p^\ell \times d} \quad \text{(표현 공간 불일치)}$$

$$\text{Lighthouse: } Q^{(\ell)}, K^{(\ell)} \in \mathbb{R}^{N/p^\ell \times d} \quad \text{(동일 표현 공간)}$$

대칭 풀링이 가능케 하는 **summary-summary interactions**는 비대칭 피라미드가 표현할 수 없는 멀티스케일 패턴을 학습하게 합니다. 이는 모델이 다양한 컨텍스트 길이에서 일반화할 수 있는 표현을 습득하는 데 기여할 수 있습니다.

### 3.4 파라미터-프리 스코어링의 일반화 이점

학습 가능한 선택기(NSA, DSA)의 문제점:
- **Scorer collapse:** 선택기가 소수 패턴에 과적합
- **Scorer-attention misalignment:** 선택 품질과 어텐션 품질의 분리
- **보조 손실 튜닝:** 추가 하이퍼파라미터

Lighthouse의 $\ell_2$ 노름 기반 파라미터-프리 스코어링은:

$$s_{0,i}^{QK} = \|Q_i\|_2, \quad s_{0,i}^{KQ} = \|K_i\|_2$$

이러한 문제를 원천적으로 회피하여 **더 안정적이고 일반화 가능한 훈련**을 지원합니다. 특히, 이 스코어는 projection 행렬이 선택에 유용한 표현을 학습하도록 암묵적으로 유도합니다.

### 3.5 Dense-SDPA 복원 후 일반화 검증

NIAH 실험에서 3/4 Lighthouse 구성이 베이스라인을 초과하거나 동등한 수준을 보였습니다. 이는 계층적 훈련이 **장거리 정보 검색 능력** 측면에서도 일반화를 저해하지 않음을 입증합니다.

### 3.6 일반화 가능성의 잠재적 확장

논문이 암시하는 일반화 가능성:

1. **멀티모달 확장:** "다중 해상도 피라미드는 비전, 오디오, 비디오로 자연스럽게 확장"
2. **더 큰 모델/토큰 예산:** $k$와 손실의 관계가 더 큰 예산에서 역전될 가능성이 있어 추가 연구 필요
3. **다양한 아키텍처 적용 가능성:** Dense-SDPA 재개 전략은 어떤 Transformer 변형에도 적용 가능

---

## 4. 미래 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

#### 패러다임 전환: 훈련-시 희소성의 재정의

기존 연구는 희소 어텐션을 추론 효율화 도구로 주로 활용했습니다. Lighthouse는 **"훈련 보조 도구"로서의 희소 어텐션**이라는 새로운 패러다임을 제시합니다:

$$\text{기존: Dense Train} \rightarrow \text{Sparse Inference}$$
$$\text{Lighthouse: Sparse Train (Lighthouse)} \rightarrow \text{Dense Inference (SDPA Resume)}$$

이는 다음 연구 방향에 직접적 영향을 미칩니다:

1. **장문 컨텍스트 사전훈련의 실용적 접근법:** 1M 토큰 수준 훈련을 32 GPU로 실현한 구체적 방법론 제시
2. **훈련-정확성 기준의 표준화:** 훈련-시 희소 방법 평가에 "SDPA 복원 가능성"이라는 명확한 기준 제공
3. **커널-독립적 설계 원칙:** 선택 로직을 어텐션 커널 외부에 배치하는 설계가 하드웨어 독립성과 미래 호환성을 크게 향상

#### 관련 분야 촉진

- **KV 캐시 관리 연구:** Dense-SDPA 복원 이후 캐시 최적화와의 결합 가능성
- **컨텍스트 병렬화 연구:** 표준 Ring Attention과의 호환성이 입증되어 다중 노드 장문 훈련 연구 활성화
- **멀티모달 모델 훈련:** 피라미드 구조가 비전/오디오/비디오의 자연적 다중 해상도에 적합

### 4.2 향후 연구 시 고려할 점

#### (1) 스케일 검증의 필요성

현재 실험은 530M 모델에 국한되어 있습니다. 더 큰 모델(7B, 70B, 수백 B 파라미터)에서:
- 계층적 훈련의 정규화 효과가 유지되는지
- 최적 하이퍼파라미터($L, p, k$)가 스케일에 따라 변하는지
- $k$와 손실의 비단조 관계가 더 큰 토큰 예산에서 어떻게 변하는지

#### (2) 적응형 Top-K 예산

현재 $k$는 모든 레이어와 헤드에서 고정됩니다. 그러나 논문 자체에서 제안하듯:

> \*"Per-layer or per-head adaptive $k$ may outperform a fixed budget"*

레이어별·헤드별로 중요도에 따라 $k$를 동적으로 할당하는 방법이 성능을 더 향상시킬 수 있습니다.

#### (3) 복원 단계의 대체 방법

현재 Dense-SDPA 복원 대신 **비대칭 희소 방법(DSA, NSA, HISA, MoBA)으로 복원**하면 추론 시에도 희소성을 유지하는 서빙 가능한 체크포인트를 얻을 수 있습니다. 이 경우:
- 추론 속도 향상 가능
- 단, 복원 단계의 정확성 보장 메커니즘 재설계 필요

#### (4) 스코어링 함수 개선

파라미터-프리 $\ell_2$ 노름 스코어는 **보수적인 하한(lower bound)** 입니다. 더 정교한 스코어링:
- QK 상호작용 기반 스코어 (O(N²/r) dilated attention의 효율적 근사)
- 학습 가능한 경량 스코어링 헤드 (scorer collapse 방지 방법과 함께)

#### (5) 다운스트림 태스크 평가

현재는 훈련 손실과 NIAH만 평가됩니다. 실용적 채택을 위해서는:
- LongBench, SCROLLS, ∞-Bench 등 장문 이해 벤치마크
- MMLU, GSM8K 등 일반 능력 벤치마크에서의 회귀(regression) 여부 검증
- 코드 생성, 멀티-홉 추론 등 복잡한 태스크에서의 성능

#### (6) 서빙 통합

> *"Serving integration (continuous batching, speculative decoding, KV-cache management) is needed to translate the training speedups into deployment."*

훈련 속도 향상이 실제 배포 이점으로 이어지려면 추론 엔진 통합이 필수적입니다.

#### (7) 훈련 안정성의 이론적 이해

Dense-SDPA 복원 시 발생하는 손실 급등(spike) 현상(1.12~1.57)의 원인과 회복 메커니즘에 대한 이론적 이해가 필요합니다. 이는 최적의 전환 시점 결정과 직결됩니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 훈련/추론 | Q 압축 | 커널 독립성 | 훈련-정확성 기준 | 참고문헌 |
|------|------|-----------|--------|------------|----------------|---------|
| **FlashAttention-2** (2023) | IO-aware Dense Attention | 양방향 | ✗ (dense) | ✓ | N/A | arXiv:2307.08691 |
| **FlashAttention-3** (2024) | 비동기 저정밀도 Dense | 양방향 | ✗ (dense) | ✓ | N/A | NeurIPS 2024 |
| **Mamba** (2023) | SSM/상태공간 | 양방향 | N/A | ✓ | N/A | arXiv:2312.00752 |
| **MoBA** (2025) | 블록 레벨 Mixture-of-Experts Attention | 훈련 포함 | ✗ | ✗ (custom) | 미검증 | arXiv:2502.13189 |
| **NSA** (2025) | 네이티브 희소 어텐션 | 훈련 포함 | ✗ | ✗ (custom) | 부분 검증 | arXiv:2502.11089 |
| **MInference** (2024) | 동적 희소 어텐션 (추론 전용) | 추론 전용 | ✗ | ✓ | 해당없음 | arXiv:2407.02490 |
| **HISA** (2026) | 계층적 인덱서 (DSA 플러그인) | 추론 전용 | ✗ | ✗ (custom) | 해당없음 | arXiv:2603.28458 |
| **H2O** (2023) | 중요 토큰 KV 캐시 축출 | 추론 전용 | ✗ | ✓ | 해당없음 | NeurIPS 2023 |
| **SnapKV** (2024) | 동적 KV 캐시 압축 | 추론 전용 | ✗ | ✓ | 해당없음 | arXiv:2404.14469 |
| **Ring Attention** (2023) | 블록와이즈 컨텍스트 병렬화 | 양방향 | ✗ (dense) | ✓ | N/A | arXiv:2310.01889 |
| **InfLLM-V2** (2026) | Dense-Sparse 전환 어텐션 | 추론 중심 | ✗ | ✗ (custom) | 부분 | arXiv (2026) |
| **Log-Linear Attention** (2025) | 로그선형 복잡도 | 양방향 | N/A | ✓ | N/A | arXiv (2025) |
| **DSA/DeepSeek** (2025) | 학습 가능한 토큰 선택 | 훈련 포함 | ✗ | ✗ (custom) | 미검증 | arXiv:2412.19437 |
| **Lighthouse** (2026) | 대칭 계층 희소 어텐션 | **훈련 전용** | **✓** | **✓** | **명시적 검증** | arXiv:2605.06554 |

### 핵심 차별화 포인트

**NSA vs. Lighthouse:**
- NSA: $Q$는 full resolution, $K/V$만 압축 → 비대칭 → FlashAttention 직접 사용 불가
- Lighthouse: $Q, K, V$ 대칭 압축 → FlashAttention을 서브시퀀스에 그대로 적용

**MoBA vs. Lighthouse:**
- MoBA: 블록 단위 선택이 커널 내부에 융합 → 커널 결합, 훈련-정확성 미검증
- Lighthouse: 선택을 커널 외부에 배치 → 커널 독립성, 명시적 정확성 검증

**HISA vs. Lighthouse:**
- HISA: 추론 전용, Dense 백본에서 정확성 상속, DSA 커널 필요
- Lighthouse: 훈련 시 희소성 → Dense 추론, 독립적 정확성 입증 필요 (입증 완료)

---

## 참고자료

**주 논문 (분석 대상):**
- Peng, B., Ghosh, S., & Quesnelle, J. (2026). *Long Context Pre-Training with Lighthouse Attention*. arXiv:2605.06554v1.

**논문 내 참조 문헌 (주요 비교 대상):**
- Dao, T. (2024). *FlashAttention-2: Faster attention with better parallelism and work partitioning*. ICLR 2024. arXiv:2307.08691.
- Shah, J. et al. (2024). *FlashAttention-3: Fast and accurate attention with asynchrony and low-precision*. NeurIPS 2024.
- Gu, A. & Dao, T. (2023). *Mamba: Linear-time sequence modeling with selective state spaces*. arXiv:2312.00752.
- Lu, E. et al. (2025). *MoBA: Mixture of block attention for long-context LLMs*. arXiv:2502.13189.
- Yuan, J. et al. (2025). *Native Sparse Attention: Hardware-aligned and natively trainable sparse attention*. arXiv:2502.11089.
- DeepSeek-AI. (2025). *DeepSeek-V3.2-Exp: Boosting long-context efficiency with DeepSeek sparse attention*. arXiv preprint.
- Zhao, L. et al. (2026). *HISA: Efficient hierarchical indexing for fine-grained sparse attention*. arXiv:2603.28458.
- Zhang, Z. et al. (2023/2024). *H2O: Heavy-hitter oracle for efficient generative inference of large language models*. NeurIPS 2023/2024. arXiv:2306.14048.
- Li, Y. et al. (2024). *SnapKV: LLM knows what you are looking for before generation*. arXiv:2404.14469.
- Liu, H., Zaharia, M., & Abbeel, P. (2023). *Ring attention with blockwise transformers for near-infinite context*. arXiv:2310.01889.
- Jiang, H. et al. (2024). *MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention*. arXiv:2407.02490.
- Zhao, L. et al. (2026). *InfLLM-V2: Dense–sparse switchable attention for seamless short-to-long adaptation*. arXiv preprint.
- Yang, Z. et al. (2016). *Hierarchical attention networks for document classification*. NAACL 2016.
- Guo, H. et al. (2025). *Log-linear attention*. arXiv preprint.
- Arora, S. et al. (2024). *Zoology: Measuring and improving recall in efficient language models*. ICLR 2024. arXiv:2312.04927.
- GitHub 코드: https://github.com/ighoshsubho/lighthouse-attention

> **주의:** 본 논문은 2026년 5월 arXiv에 제출된 프리프린트(arXiv:2605.06554v1)로, 아직 동료심사(peer review)를 거치지 않았습니다. 530M 소규모 실험에 기반한 예비적(preliminary) 결과임을 감안해야 합니다.
