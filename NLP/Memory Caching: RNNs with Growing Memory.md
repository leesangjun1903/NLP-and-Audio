# Memory Caching: RNNs with Growing Memory 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문(Behrouz et al., arXiv:2602.24281v1, 2026)은 **순환 신경망(RNN)의 고정 크기 메모리가 장문 맥락 및 회상 집약적(recall-intensive) 태스크에서의 성능 병목**임을 지적하고, 이를 해결하기 위한 **Memory Caching(MC)** 기법을 제안합니다.

> **핵심 주장**: RNN의 메모리 상태(hidden state)의 체크포인트를 캐싱(caching)함으로써, RNN의 유효 메모리 용량을 시퀀스 길이에 따라 성장시킬 수 있으며, 이는 $\mathcal{O}(L)$ 복잡도(RNN)와 $\mathcal{O}(L^2)$ 복잡도(Transformer) 사이의 유연한 절충점을 제공한다.

### 주요 기여 세 가지

| 기여 항목 | 내용 |
|---|---|
| **MC 프레임워크** | 시퀀스를 세그먼트로 분할하고 각 세그먼트의 압축된 메모리 상태를 캐싱 |
| **4가지 집계 전략** | Residual Memory, Gated Residual Memory(GRM), Memory Soup, Sparse Selective Caching(SSC) |
| **실증적 검증** | LA, DLA, SWLA, Titans 아키텍처에 MC 적용, 언어 모델링·장문 이해·회상 태스크에서 성능 향상 확인 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

**Transformer의 문제**: Attention 메커니즘은 과거 모든 토큰을 캐싱하는 성장하는 메모리(growing memory)를 가지므로 $\mathcal{O}(L^2)$ 복잡도와 높은 KV-캐시 메모리를 요구합니다.

**RNN의 문제**: 고정 크기 메모리(fixed-size memory)로 시퀀스를 압축하기 때문에, 시퀀스가 길어질수록 과거 정보를 **망각(forget)**하게 됩니다. 이는 회상 집약적 태스크와 장문 맥락 태스크에서 치명적입니다.

$$\text{표준 Attention: } \mathbf{y}_i = \frac{1}{Z_i}\sum_{t=1}^{i} \exp\left(\mathbf{q}_i^\top \mathbf{k}_t\right)\mathbf{v}_t \quad \cdots\mathcal{O}(L^2)$$

$$\text{Linear Attention: } \mathbf{y}_i = \frac{1}{Z_i}\mathcal{M}_i\phi(\mathbf{q}_i), \quad \mathcal{M}_t = \mathcal{M}_{t-1} + \mathbf{v}_t\phi(\mathbf{k}_t)^\top \quad \cdots\mathcal{O}(L)$$

---

### 2.2 제안하는 방법 (수식 포함)

#### 기본 MC 프레임워크

시퀀스 $x \in \mathbb{R}^{L \times d_{in}}$을 $N$개의 세그먼트 $S^{(1)}, \ldots, S^{(N)}$으로 분할하고, 각 세그먼트에 대한 메모리를 업데이트한 뒤 마지막 상태를 캐싱합니다.

$$\mathbf{k}_t = x_t W_k, \quad \mathbf{v}_t = x_t W_v, \quad \mathbf{q}_t = x_t W_q$$

$$\mathcal{M}_t^{(s)} = f\!\left(\mathcal{M}_{t-1}^{(s)};\, \mathbf{k}_t, \mathbf{v}_t\right), \quad \text{where } 1 \le t \le L^{(s)} $$

출력 계산 시, 현재 메모리(online memory)와 과거 캐싱된 메모리를 함께 활용합니다:

```math
\mathbf{y}_t = \text{Agg}\!\left(\left\{\mathcal{M}_{L^{(1)}}^{(1)}(\cdot),\ldots,\mathcal{M}_{L^{(s-1)}}^{(s-1)}(\cdot)\right\};\, \mathcal{M}_t^{(s)}(\cdot);\, \mathbf{q}_t\right)
```

---

#### 변형 1: Residual Memory

가장 단순한 형태로, 캐싱된 메모리들을 **잔차 연결(residual connection)** 방식으로 합산합니다:

$$\mathbf{y}_t = \underbrace{\mathcal{M}_t^{(s)}(\mathbf{q}_t)}_{\text{Online Memory}} + \underbrace{\sum_{i=1}^{s-1}\mathcal{M}_{L^{(i)}}^{(i)}(\mathbf{q}_t)}_{\text{Cached Memories}} $$

---

#### 변형 2: Gated Residual Memory (GRM)

입력 의존적 게이팅(gating) 파라미터 $\gamma_t^{(i)}$를 도입하여 각 세그먼트의 기여도를 **선택적으로 조절**합니다:

$$\mathbf{y}_t = \gamma_t^{(s)}\mathcal{M}_t^{(s)}(\mathbf{q}_t) + \sum_{i=1}^{s-1}\gamma_t^{(i)}\mathcal{M}_{L^{(i)}}^{(i)}(\mathbf{q}_t) $$

게이팅 파라미터는 현재 토큰의 입력과 세그먼트 문맥의 유사도로 정의됩니다:

$$\gamma_t^{(i)} = \langle \mathbf{u}_t,\, \text{MeanPooling}(S^{(i)})\rangle, \quad \mathbf{u}_t = x_t W_u $$

---

#### 변형 3: Memory Soup

Wortsman et al. (2022)의 **모델 수프(model soup)** 아이디어에서 영감을 받아, 캐싱된 메모리 모듈의 **파라미터 자체를 가중 평균**하여 새로운 메모리를 구성합니다:

```math
\boldsymbol{\theta}_{\mathcal{M}_t^*} := \left\{\sum_{i=1}^{s}\gamma_t^{(i)}W_1^{(i)},\ldots,\sum_{i=1}^{s}\gamma_t^{(i)}W_c^{(i)}\right\}
```

$$\mathbf{y}_t = \mathcal{M}_t^*(\mathbf{q}_t) $$

> 선형 메모리에서는 GRM과 수학적으로 동치이지만, **비선형(deep) 메모리**에서는 차별화된 표현력을 발휘합니다.

---

#### 변형 4: Sparse Selective Caching (SSC)

MoE(Mixture of Experts) 스타일의 **라우터**를 이용해 각 토큰에 가장 관련성 높은 $k$개의 캐싱 메모리만 선택합니다:

$$r_t^{(i)} = \langle \mathbf{u}_t,\, \text{MeanPooling}(S^{(i)})\rangle, \quad \mathbf{u}_t = x_t W_u $$

$$\mathcal{R}_t = \arg\text{Top-}k\!\left(\{r_t^{(i)}\}_{i=1}^{s-1}\right)$$

$$\mathbf{y}_t = \gamma_t^{(s)}\mathcal{M}_t^{(s)}(\mathbf{q}_t) + \sum_{i \in \mathcal{R}_t}\gamma_t^{(i)}\mathcal{M}_{L^{(i)}}^{(i)}(\mathbf{q}_t) $$

---

#### 복잡도 비교

| 방법 | 복잡도 |
|---|---|
| 표준 RNN | $\mathcal{O}(L)$ |
| Transformer | $\mathcal{O}(L^2)$ |
| Memory Caching (MC) | $\mathcal{O}(NL)$, $\;1 \le N \le L$ |
| 등 크기 세그먼트 (크기 $C$) | $\mathcal{O}\!\left(p \cdot \frac{L^2}{C}\right)$ |
| 로그 분할 세그먼트 | $\mathcal{O}(p \cdot L\log L)$ |

---

### 2.3 모델 구조

논문은 MC를 세 가지 베이스 아키텍처에 적용합니다:

**1. Linear Attention (LA)** (Katharopoulos et al., 2020):

$$\mathcal{M}_t^{(s)} = \mathcal{M}_{t-1}^{(s)} + \mathbf{v}_t\mathbf{k}_t^\top$$

$$\mathbf{y}_t = \left(\mathcal{M}_t^{(s)} + \sum_{i=1}^{s-1}\mathcal{M}_{L^{(i)}}^{(i)}\right)\mathbf{q}_t $$

**2. Sliding Window Linear Attention (SWLA)** (Behrouz et al., 2025a):

$$\mathcal{M}_t^{(s)} = \alpha_t\mathcal{M}_{t-1}^{(s)} + \left(\beta_t\mathbf{v}_{t-1}\mathbf{k}_{t-1}^\top + \lambda_t\mathbf{v}_t\mathbf{k}_t^\top\right) $$

**3. Deep Linear Attention (DLA)** (Behrouz et al., 2025a):

$$\mathcal{M}_t^{(s)} = \mathcal{M}_{t-1}^{(s)} - \eta_t\nabla\mathcal{L}\!\left(\mathcal{M}_{t-1}^{(s)};\, \mathbf{k}_t, \mathbf{v}_t\right) $$

**4. Titans (LMM)** (Behrouz et al., 2025c):

$$\mathcal{M}_t = \alpha_t\mathcal{M}_{t-1} - S_t$$

$$S_t = \beta_t S_{t-1} - \eta_t\nabla\mathcal{L}(\mathcal{M}_{t-1};\mathbf{k}_t, \mathbf{v}_t), \quad \mathcal{L} = \|\mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2 $$

---

### 2.4 성능 향상

**언어 모델링 및 상식 추론 (Table 1)**:

| 모델 | Wiki ppl↓ | LMB ppl↓ | 평균 acc↑ |
|---|---|---|---|
| Titans (1.3B) | 15.60 | 11.41 | 56.82 |
| Titans + GRM | **15.37** | **11.29** | **58.33** |
| DLA (1.3B) | 16.31 | 12.29 | 53.72 |
| DLA + GRM | **16.08** | **12.10** | **55.96** |

**Needle-In-A-Haystack (Table 2)**:

- DLA 단독: S-NIAH-1 16K에서 44.0%
- DLA + GRM: S-NIAH-1 16K에서 **82.4%** (거의 두 배)
- Titans + GRM: S-NIAH-2 16K에서 **88.2%** (Titans 단독 75.4% 대비 크게 향상)

**LongBench 장문 이해 (Table 4)**: 모든 MC 변형이 기본 RNN 대비 성능 향상 확인

---

### 2.5 한계

1. **Transformer와의 격차**: 회상 집약적 인-컨텍스트 태스크(Table 3)에서 Transformer는 여전히 최고 성능을 보이며, MC 변형은 이를 완전히 따라잡지 못합니다.
2. **메모리 오버헤드**: 모든 캐싱 메모리를 유지할 경우 극장문 시퀀스에서 메모리 부담이 증가합니다. (SSC로 부분 완화)
3. **세그먼테이션 설계의 민감성**: 세그먼트 크기는 압축 수준과 계산 비용 간의 트레이드오프를 결정하며, 최적 선택이 태스크 의존적입니다.
4. **풀링 메커니즘의 단순성**: MeanPooling은 세그먼트 표현을 단순화하므로, 더 표현력 있는 풀링/라우팅 메커니즘으로 개선 여지가 있습니다.
5. **로그 분할의 한계**: 로그 분할은 효율적이지만 초기 긴 서브시퀀스에서 메모리 오버플로우를 일으킬 수 있고, 짧은 서브시퀀스에서는 메모리가 충분히 최적화되지 않을 수 있습니다.
6. **선형 메모리에서의 붕괴**: Residual Memory는 선형 메모리 모듈에 적용할 경우 수학적으로 고정 크기 메모리로 붕괴됩니다(Equation 13 참조).

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

**유효 메모리 용량의 성장**이 일반화 향상의 근본 원인입니다. MC는 RNN이 과거 전체 히스토리에 압축된 형태로 직접 접근할 수 있게 하여, 학습 중 보지 못한 긴 시퀀스에도 일반화가 가능합니다.

#### (a) 포스트 학습 적용 가능성 (Post-Training Application)

논문은 MC가 **사전 학습 이후에도 적용 가능**함을 보입니다:

> "Memory caching can also be applied after pre-training of the model, where at inference, we cache the state of the memory after each segment... even this simple technique can enhance the length extrapolation capability of recurrent models significantly."

이는 **파인튜닝 없이도 기존 모델의 일반화 범위를 확장**할 수 있음을 의미하며, 실용적 관점에서 매우 중요합니다.

#### (b) 컨텍스트 의존적 게이팅과 일반화

GRM의 핵심인 게이팅 파라미터 $\gamma_t^{(i)}$는:

$$\gamma_t^{(i)} = \text{softmax}\!\left(\langle x_t W_u,\, \text{MeanPooling}(S^{(i)})\rangle\right)$$

단순한 위치 기반 필터링이 아닌 **쿼리와 과거 세그먼트 문맥 간의 유사도**에 기반하여 선택적으로 정보를 검색합니다. 이는 모델이 학습 시 보지 못한 새로운 시퀀스 패턴에도 **관련 과거 정보를 적응적으로 활용**할 수 있게 합니다.

Ablation Study (Table 5)에서:
- Context-dependent 게이팅 제거 시: retrieval acc가 40.5% → 33.0%로 크게 하락
- 이는 컨텍스트 의존성이 일반화 핵심 요소임을 시사

#### (c) 메모리 아키텍처 표현력과 견고성

> "Surprisingly, using memory caching results in more robustness of the performance with respect to the memory architecture and expressivity."

즉, **선형 메모리**와 같이 표현력이 낮은 경우에도 MC를 적용하면 비선형 메모리 모듈에 근접하는 성능을 보이는 것으로, MC 자체가 **아키텍처 표현력 부족을 보완**하는 정규화(regularization) 효과를 가짐을 시사합니다.

#### (d) SSC의 희소 선택적 일반화

SSC는 관련 없는 과거 메모리의 노이즈를 줄이고 **가장 관련성 높은 정보만 선택적으로 활용**합니다. 이는 Mixture-of-Experts의 희소성 원리와 유사하게, 모델이 **태스크 관련 특징에 집중**하도록 유도하여 일반화를 향상시킵니다.

#### (e) 네스티드 학습 관점(Nested Learning Perspective)에서의 일반화

논문은 Behrouz et al. (2026)의 TTM(Test-Time Memorization) 프레임워크를 활용하여, 메모리를 **연속적인 최적화 과정의 체크포인트**로 해석합니다:

$$\mathcal{M}_{t+1} = \arg\min_{\mathcal{M}}\,\mathcal{L}(\mathcal{M}(\mathbf{k}_t);\mathbf{v}_t) + \text{Ret}(\mathcal{M};\mathcal{M}_t) $$

이 관점에서 캐싱된 메모리는 최적화 경로의 중간 해(intermediate solution)이며, 이를 활용하면 모델이 **더 풍부한 최적화 경로 정보**에 기반하여 일반화할 수 있습니다.

#### (f) MQAR 및 다양한 벤치마크에서의 일반화 검증

MQAR(Multi-Query Associative Recall) 실험(Figure 5)에서 MC 변형들은 다양한 차원(dimension) 값에 걸쳐 기본 RNN 대비 일관되게 높은 성능을 보이며, 이는 **특정 태스크 설정에 과적합되지 않은 일반적인 개선**임을 시사합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (a) RNN–Transformer 연속체(Continuum) 설계 패러다임 정립

MC는 $N=1$(순수 RNN)에서 $N=L$(Transformer)까지 연속적으로 보간할 수 있는 **통합 설계 공간**을 제시합니다. 이는 미래의 아키텍처 탐색이 이분법적(either RNN or Transformer)이 아닌 **연속적 공간에서 이루어질 수 있음**을 시사합니다.

#### (b) 하이브리드 모델에 대한 이론적 정당화

논문은 MC가 단순화된 설정에서 하이브리드 모델(Attention + RNN)과 수학적으로 동치임을 보입니다:

$$\mathbf{y}_t = \left(\sum_{i=1}^{t}\frac{\exp(\mathbf{u}_t^\top\mathbf{k}_i)}{\sum_{\ell=1}^{i}\exp(\mathbf{u}_t^\top\mathbf{k}_\ell)}\mathbf{v}'_i\right) \otimes \sigma(x_t W_Q) $$

이는 하이브리드 모델의 성능 이유를 **메모리 용량의 확장**으로 설명하는 **이론적 프레임워크**를 제공하며, 향후 하이브리드 아키텍처 설계 연구의 이론적 토대가 됩니다.

#### (c) 포스트-학습 컨텍스트 확장 기법 개발 촉진

MC를 사전 학습 후 inference 단계에 적용 가능함을 보임으로써, **사전 학습된 RNN의 컨텍스트 길이를 재학습 없이 확장**하는 연구 방향을 열었습니다.

#### (d) 딥 메모리 모듈과의 결합 가능성

MC는 선형 메모리와 비선형(deep) 메모리 모두에 적용 가능하며, 특히 딥 메모리에서는 **선형 메모리와 근본적으로 다른 새로운 아키텍처 클래스**를 생성합니다. 이는 MLP 기반 메모리 모듈의 새로운 설계 가능성을 열어줍니다.

#### (e) 메모리를 최적화 체크포인트로 보는 관점의 확산

TTM/Miras 프레임워크와의 결합을 통해, **메모리 상태를 최적화 과정의 체크포인트로 해석**하는 패러다임이 더 넓은 시퀀스 모델링 연구에 영향을 미칠 것으로 예상됩니다.

---

### 4.2 향후 연구 시 고려할 점

#### (a) 더 표현력 있는 세그먼트 표현 방법

현재 MeanPooling은 세그먼트 문맥을 지나치게 단순화합니다. 향후 연구에서는 다음과 같은 더 강력한 표현 방법을 고려해야 합니다:
- **계층적 표현**: 세그먼트 내 토큰들의 계층적 집계
- **학습 가능한 풀링**: 어텐션 기반 풀링 또는 학습된 집계 함수
- **SSM 기반 요약**: 각 세그먼트를 별도의 SSM으로 요약

#### (b) 동적 세그먼테이션 전략

현재 고정 크기 또는 로그 분할 방식은 입력 내용을 고려하지 않습니다. 향후 연구에서는:
- **내용 기반 동적 세그먼테이션**: 문장 경계, 의미 단위 등을 고려한 분할
- **학습 가능한 세그먼테이션**: 세그먼트 경계를 모델이 학습하도록 설계

#### (c) 메모리 압축과 선택의 공동 최적화

SSC의 라우터와 캐싱 메모리를 **엔드-투-엔드로 공동 학습**하는 방법을 개발해야 합니다. 현재 MeanPooling 기반 라우터는 학습 과정에서 최적화되지 않을 수 있습니다.

#### (d) 스케일링 법칙 분석

MC 변형들의 **스케일링 특성**(파라미터 수, 시퀀스 길이, 세그먼트 수에 따른 성능 변화)을 체계적으로 분석해야 합니다. 특히 캐싱되는 세그먼트 수 $N$과 성능 간의 관계를 명확히 밝혀야 합니다.

#### (e) 추론 효율성 최적화

학습 시에는 세그먼트별 메모리를 순차적으로 처리하지만, **추론 시 병렬화** 방법과 캐싱 메모리의 효율적 저장·로딩 전략이 필요합니다. SSC가 GPU/TPU 메모리 절약 가능성을 보였으나, 실제 시스템 수준의 최적화 연구가 필요합니다.

#### (f) 이론적 보장 및 분석

MC가 제공하는 **표현력의 이론적 상한**, 망각률 감소에 대한 정보 이론적 분석, 그리고 일반화 오차 경계(generalization bound)에 대한 수학적 분석이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 장문 컨텍스트 처리 관련 주요 연구 비교

| 연구 | 방법 | 복잡도 | MC와의 관계 |
|---|---|---|---|
| **Transformer-XL** (Dai et al., 2019) | 이전 세그먼트 hidden state를 캐싱하여 컨텍스트 확장 | $\mathcal{O}(L^2)$ | MC의 개념적 선조; MC는 이를 RNN에 일반화 |
| **Infini-Attention** (Munkhdalai et al., 2024) | 압축 메모리와 로컬 어텐션 결합 | 선형 | 유사한 동기; MC는 더 다양한 집계 전략 제공 |
| **Log-Linear Attention** (Guo et al., 2025) | Fenwick 트리 기반 로그 선형 숨겨진 상태 | $\mathcal{O}(L\log L)$ | MC의 Log-Linear++ 변형으로 직접 비교; 고정 분할의 한계를 MC가 극복 |
| **MoBA** (Lu et al., 2025) | 시퀀스 차원에서 MoE 스타일 블록 어텐션 | $\mathcal{O}(L^2/C)$ | MC와 유사한 블록화; MC는 어텐션 기반이 아닌 RNN 기반으로 차별화 |
| **TTT (Test-Time Training)** (Sun et al., 2024) | $L_2$ 회귀 기반 테스트 시간 가중치 업데이트 | $\mathcal{O}(L)$ | MC의 기저 이론(TTM 프레임워크)과 연관; MC는 메모리 체크포인팅 추가 |
| **Titans** (Behrouz et al., 2025c) | 딥 메모리 모듈 + 모멘텀 최적화 | $\mathcal{O}(L)$ | MC의 주요 베이스라인; MC 적용 시 가장 큰 성능 향상 |
| **Atlas** (Behrouz et al., 2025a) | Omega 학습 규칙 + Muon 최적화 | $\mathcal{O}(L)$ | DLA의 확장; MQAR에서 MC 변형과 비교됨 |
| **RWKV-7** (Peng et al., 2024) | 행렬 값 상태 + 동적 재귀 | $\mathcal{O}(L)$ | MC 베이스라인; MC 적용 가능성 미탐색 |
| **Samba** (Ren et al., 2024) | Attention + Linear RNN 하이브리드 | 혼합 | MC의 동기와 유사; MC는 순수 RNN에서 하이브리드 효과 달성 |
| **DeltaNet/DeltaProduct** (Yang et al., 2024c; Siems et al., 2025) | 델타 규칙 기반 업데이트 | $\mathcal{O}(L)$ | MC 베이스라인으로 사용 가능; 표현력 향상을 다른 방향으로 추구 |
| **Miras/Memora** (Behrouz et al., 2026) | TTM 통합 프레임워크 | $\mathcal{O}(L)$ | MC의 이론적 기반; MC는 Miras에 캐싱 기능을 추가 |

### 5.2 핵심 차별점 분석

```
[RNN 계열]              [하이브리드 계열]         [Transformer 계열]
  Titans                   Samba                  Transformer++
  DLA, SWLA          →    MoBA, Infini-Attn  ←   Log-Linear Attn
  RWKV-7                   MC (제안)               (Gated) Attention

복잡도: O(L)           O(NL), N∈[1,L]              O(L²)
표현력: 낮음            중간~높음                    높음 (회상 태스크)
```

**MC의 핵심 차별점**:
1. **아키텍처 비의존성**: 어떠한 RNN 업데이트 규칙에도 적용 가능
2. **연속적 복잡도 제어**: $N$을 조절하여 복잡도를 연속적으로 제어
3. **딥 메모리 지원**: 비선형 메모리 모듈에서 새로운 아키텍처 클래스 생성
4. **포스트 학습 적용**: 사전 학습된 모델에 추론 시 적용 가능

---

## 참고 자료

**주 논문**:
- Behrouz, A., Li, Z., Deng, Y., Zhong, P., Razaviyayn, M., & Mirrokni, V. (2026). *Memory Caching: RNNs with Growing Memory*. arXiv:2602.24281v1.

**논문 내 인용 핵심 참고문헌**:
- Vaswani, A. et al. (2017). *Attention is All You Need*. NeurIPS.
- Katharopoulos, A. et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.
- Behrouz, A., Zhong, P., & Mirrokni, V. (2025c). *Titans: Learning to Memorize at Test Time*. NeurIPS 2025.
- Behrouz, A. et al. (2025a). *Atlas: Learning to Optimally Memorize the Context at Test Time*. arXiv:2505.23735.
- Behrouz, A. et al. (2026). *It's All Connected: A Journey through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*. ICLR 2026.
- Wortsman, M. et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time*. ICML.
- Guo, H. et al. (2025). *Log-Linear Attention*. arXiv:2506.04761.
- Sun, Y. et al. (2024). *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*. arXiv:2407.04620.
- Lu, E. et al. (2025). *MoBA: Mixture of Block Attention for Long-Context LLMs*. NeurIPS 2025.
- Munkhdalai, T. et al. (2024). *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-Attention*. arXiv:2404.07143.
- Ren, L. et al. (2024). *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*. arXiv:2406.07522.
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
- Arora, S. et al. (2024b). *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff*. ICML.
- Bai, Y. et al. (2024). *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*. ACL.
- Merrill, W. et al. (2024). *The Illusion of State in State-Space Models*. ICML.

> **정확도 관련 고지**: 본 답변은 제공된 PDF 원문(arXiv:2602.24281v1)에 근거하여 작성되었으며, 논문 내에 명시되지 않은 내용(예: 2020년 이후 타 논문의 구체적 수치)은 논문의 Related Work 및 인용 목록에 기반하여 기술하였습니다. 논문 자체가 arXiv 프리프린트(2026년 2월 27일 제출)임을 감안하여, 일부 인용 논문(2025~2026년)은 아직 미발표 또는 출판 전 단계일 수 있음을 유의하시기 바랍니다.
