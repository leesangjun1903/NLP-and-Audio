# Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **"메모리 병목(memory bottleneck)이 아닌 연산 병목(computation bottleneck)이 SSM 기반 하이브리드 모델의 추론 실패 원인이며, 생물학적 수면(sleep)에서 영감을 받은 오프라인 반복 연산(offline recurrence)을 통해 이를 해결할 수 있다."**

즉, 기존의 Attention-SSM 하이브리드 모델이 긴 문맥 처리에서 실패하는 이유는 단순히 fast weight의 용량(capacity) 부족이 아니라, **evicted된 context를 유용한 내부 표현으로 변환하는 데 필요한 연산량의 부족**이라는 점을 주장합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **문제 재정의** | SSM 하이브리드 모델의 실패 원인을 memory capacity가 아닌 computation depth 부족으로 규명 |
| **LLM Sleep 메커니즘** | 컨텍스트 윈도우가 꽉 찰 때 $N$번의 오프라인 재귀 패스를 수행하는 sleep 메커니즘 제안 |
| **예측 지연 보존** | Sleep 단계에 추가 연산을 집중시켜 예측(wake) 단계의 단일 forward pass latency 유지 |
| **합성 벤치마크 검증** | Rule 110 Cellular Automaton, Depo (multi-hop graph retrieval) 과제에서 효과 검증 |
| **실제 LLM 적용 검증** | GSM-Infinite에서 사전학습된 Jet-Nemotron 2B, Ouro 1.4B 모델에 적용하여 효과 확인 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**트랜스포머의 확장성 문제:**

- Softmax Attention의 계산 복잡도는 컨텍스트 길이 $T$에 대해 $O(T^2)$으로 증가
- KV 캐시의 메모리는 $O(T)$으로 선형 증가

**기존 해법의 한계:**

SSM(State Space Model) 기반 하이브리드 모델은 fast weight $\mathbf{S}_t$를 통해 이를 완화하려 했으나, 논문은 이 모델들이 **추론 깊이(reasoning depth)가 증가할수록 성능이 급격히 저하**됨을 보입니다. 이때 핵심은 메모리 용량이 고정되어도 이 문제가 발생한다는 것입니다.

**구체적 실패 사례 (Rule 110 Cellular Automaton):**

- 4개의 이진 상태($T = 96$ tokens)를 처리한 후, $t$번 전이된 각 상태의 첫 번째 비트를 예측
- 롤아웃 깊이 $t$가 증가할수록 4-layer GDN-Attention hybrid 모델의 정확도가 급감
- 이는 추론 깊이 문제이지 메모리 용량 문제가 아님

### 2.2 제안하는 방법 (수식 포함)

#### Attention의 기본 연산

$$\boldsymbol{q}_t = \mathbf{W}_Q \boldsymbol{x}_t, \quad \boldsymbol{k}_t = \mathbf{W}_K \boldsymbol{x}_t, \quad \boldsymbol{v}_t = \mathbf{W}_V \boldsymbol{x}_t \tag{1}$$

$$\boldsymbol{o}_t = \mathbf{V}_t^\top \text{softmax}\!\left(\frac{\mathbf{K}_t \boldsymbol{q}_t}{\sqrt{d}}\right) \tag{2}$$

여기서 KV 캐시 $\mathbf{K}_t, \mathbf{V}_t$는 시퀀스 길이에 비례하여 선형 증가합니다.

#### SSM (Fast Weight) 업데이트 규칙 (Mamba2/GDN 스타일)

$$\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top, \quad \boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t \tag{3}$$

- $\alpha_t \in (0, 1)$: 데이터 의존적 forget gate
- $\beta_t \in (0, 1)$: 데이터 의존적 input gate
- $\mathbf{S}_t$는 고정 크기로 유지되어 메모리 효율적이지만, 정보 손실 발생

#### LLM Sleep: 오프라인 재귀 통합 (핵심 제안)

전체 아키텍처를 $N$번 반복(loop)하는 구조:

$$\text{Embed} \to \left[\mathcal{B}_0^{\text{attn}} \to \mathcal{B}_1^{\text{ssm}} \to \cdots \to \mathcal{B}_{D-1}^{\text{attn}}\right]^{\times N} \to \text{OutProj} \tag{6}$$

- $N = 1$일 때: 기존 vanilla SSM-Attention 하이브리드 모델과 동일
- $N > 1$일 때: Sleep 단계에서 $N$번의 재귀 패스로 fast weight $\mathbf{S}$를 반복 정제

**학습 알고리즘 (Algorithm 1 요약):**

```
입력: 토큰 x, 손실 마스크 m, 윈도우 크기 L, sleep 횟수 N
1. SSM fast weight S를 0으로 초기화
2. x, m을 길이 L의 청크로 분할
3. 각 청크 c에 대해:
   - 통합 단계 (m=0): N번 반복하여 h, S ← Blocks(h, S)
   - 예측 단계 (m≠0): 단 1회 h, S ← Blocks(h, S), 손실 계산
4. 손실 역전파 및 옵티마이저 스텝
```

**학습 손실:**

$$\mathcal{L} = \text{MaskedCE}(\text{OutProj}(\boldsymbol{h}), c, m_c)$$

- 예측 단계 토큰에만 masked cross-entropy loss 적용
- 전체 계산 그래프(통합 + 예측)를 통해 end-to-end 역전파

### 2.3 모델 구조

```
[통합 단계 (Consolidation Phase)]
컨텍스트 윈도우가 꽉 차면:
  ┌─────────────────────────────┐
  │  SSM Block  → ATTN Block   │
  │  SSM Block  → ATTN Block   │  × N (sleep loops)
  │         ...                │
  └─────────────────────────────┘
  KV 캐시 완전 초기화 (eviction)

[예측 단계 (Prediction Phase)]
  단일 forward pass → OutProj → 답 생성
```

**핵심 설계 원칙:**
- **Wake-time latency 보존**: 예측 시에는 반드시 단일 forward pass만 허용 (chain-of-thought 금지)
- **Sleep-time 연산 집중**: 추가 연산을 모두 통합 단계에 위치시킴
- **Gradient 흐름 경로**: 기존 depth-recurrent 모델과 달리, 재귀적으로 정제된 feature vector가 아닌 **정제된 fast weight $\mathbf{S}$를 통해 gradient 흐름** (sleep 후 feature는 폐기)

### 2.4 성능 향상 실험 결과

#### Task 1: Rule 110 Cellular Automaton ($t = 32$)

| 모델 | 정확도 |
|------|--------|
| No loop (baseline) | ~10% (랜덤 수준) |
| 2 loops | ~20% |
| 3 loops | >30% |
| 4 loops | >30% |

- 5B 학습 토큰 기준
- 동일 컨텍스트 길이, eviction 규칙, 예측 단계 유지 하에서 성능 향상 → 순수하게 통합 시간 연산량 증가 효과

#### Task 2: Depo (Multi-hop Graph Retrieval)

- $k$ (hop 수)가 클수록 루프 수 효과 두드러짐
- 4-hop 이상: 1-loop 모델은 거의 진전 없음
- 16-hop: 오직 4-loop 모델만 학습 시작

#### Task 3: GSM-Infinite (수학 추론, 사전학습 LLM)

**Jet-Nemotron 2B:**

| 문제 난이도 | No loop | 6 loops | 향상률 |
|------------|---------|---------|--------|
| 6 operations | 0.742 | 0.812 | **+9%** |
| 8 operations | 0.351 | 0.388 | **+11%** |

**Ouro 1.4B:**

| 문제 난이도 | No loop | 4 loops | 향상률 |
|------------|---------|---------|--------|
| 6 operations | 0.419 | 0.615 | **+47%** |
| 8 operations | 0.210 | 0.272 | **+30%** |

#### Task 4: Sliding-window Eviction (Ouro 1.4B, $L = 512$)

| 문제 난이도 | No loop | 4 loops | 향상률 |
|------------|---------|---------|--------|
| 2 operations | 0.596 | 0.905 | **+52%** |
| 4 operations | 0.839 | 0.926 | +10% |
| 6 operations | 0.251 | 0.320 | +27% |
| 8 operations | 0.116 | 0.137 | +18% |

### 2.5 한계점

| 한계 | 설명 |
|------|------|
| **학습 비용 선형 증가** | 훈련 처리량(throughput)이 $N$에 반비례하여 감소 |
| **학습 불안정성** | 깊은 재귀적 역전파로 인한 학습 불안정 가능성 |
| **순차적 학습 의존성** | 청크 $j+1$은 청크 $j$의 sleep 완료 후 처리 가능 → 시퀀스 축 완전 병렬화 불가 |
| **실험 규모 제한** | 주로 소규모 합성 과제 및 1-2B 규모 모델에서 검증 |
| **루프 수 최적화 미제시** | 최적 $N$ 선택 기준이 불명확 |
| **일반 텍스트 평가 미흡** | web-text 등 일반 perplexity 평가 없음 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 관련 핵심 메커니즘

논문이 제시하는 일반화 향상의 핵심 메커니즘은 **"query-agnostic representation 형성"** 입니다.

**Depo 태스크에서의 일반화:**

Depo 태스크에서 모델은 다음 조건 하에서 일반화해야 합니다:
- 방향 그래프 엣지가 무작위 순서로 제시됨
- 질의의 시작 노드와 홉 수 $k$가 랜덤하게 샘플링됨
- **즉, 통합 단계에서는 어떤 질의가 올지 모르는 상태에서 query-independent한 범용 표현을 형성해야 함**

이는 단순 암기(memorization)가 아닌 진정한 의미의 일반화가 필요합니다.

**GSM-Infinite에서의 분포 내 일반화:**

$$\text{훈련 분포} = \text{평가 분포 (동일 절차적 생성 방식)}$$

Kabra et al. (2026) [33]이 보인 바와 같이, GSM-Infinite와 같은 합성 데이터에서의 학습은 실제 multi-hop 추론 능력 향상으로 전이될 수 있습니다.

### 3.2 일반화를 지지하는 실험적 증거

**1. 추론 깊이 스케일링:**

$$\text{성능 향상} \propto \text{문제의 순차적 추론 깊이}$$

루프 수를 늘릴수록 어려운 인스턴스에서 더 큰 향상이 나타납니다. 이는 모델이 단순히 훈련 데이터를 암기하는 것이 아니라, 연산 능력 자체가 향상됨을 시사합니다.

**2. 사전학습 LLM에서의 효과:**

Jet-Nemotron 2B와 Ouro 1.4B는 이미 대규모 코퍼스로 사전학습된 모델입니다. 이 모델들에 sleep fine-tuning을 적용했을 때도 효과가 나타나는 것은, sleep이 새로운 지식을 주입하는 것이 아니라 **기존 지식을 더 깊이 활용하는 계산 능력을 부여**함을 보여줍니다.

**3. Distractor 토큰 하에서의 선택적 통합:**

GSM-Infinite 실험에서 질문을 컨텍스트 앞에 배치함으로써:

$$\text{질문 선 제시} \to \text{관련 정보 선택적 통합} \to \text{예측}$$

모델이 filler 토큰을 무시하고 질의와 관련된 정보를 선택적으로 통합하는 능력을 보입니다. 이는 특정 형태의 문맥 내 일반화(in-context generalization)입니다.

### 3.3 일반화 가능성의 이론적 배경

**Depth-recurrent 모델의 표현력 이론:**

Merrill and Sabharwal (2025) [40]이 보인 바와 같이:

$$\text{log-depth transformer} \supseteq \text{fixed-depth transformer (표현력)}$$

이 논문은 이 원리를 memory consolidation에 확장합니다:

$$\text{N-loop sleep} \supseteq \text{1-pass SSM (메모리 통합 표현력)}$$

**순차적 연산의 필요성:**

논문이 인용하는 Liu et al. (2025) [37]의 "Serial Scaling Hypothesis"에 따르면, 많은 추론 과제의 해는 본질적으로 순차적(sequential)이며, 이를 완전 병렬 연산으로 해결하려는 시도는 취약한 shortcut 해(brittle shortcut solutions)를 유도합니다. Sleep은 이 순차성을 자연스럽게 구현합니다.

### 3.4 일반화의 한계와 미해결 문제

**분포 외(out-of-distribution) 일반화:**

논문은 주로 동일 분포 내 일반화(in-distribution generalization)를 평가합니다. 예를 들어:
- Rule 110에서 훈련 $t$와 다른 $t$로의 일반화 미평가
- Depo에서 훈련 홉 수 범위 밖의 $k$에 대한 일반화 미평가

**일반 언어 능력 유지 여부:**

Sleep fine-tuning이 모델의 일반적인 언어 능력(general linguistic capability)에 미치는 영향은 측정되지 않았습니다. 특정 과제에 특화된 fine-tuning으로 인한 catastrophic forgetting 가능성이 존재합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

**① 메모리-연산 이분법의 재정립**

기존 연구는 긴 컨텍스트 처리의 병목을 주로 **메모리 용량** 관점에서 접근했습니다 (Arora et al. 2024 [2], Jelassi et al. 2024 [32]). 이 논문은 **연산 깊이**가 독립적인 병목임을 입증함으로써, 향후 아키텍처 설계 방향에 새로운 차원을 추가합니다.

**② Test-Time Compute 패러다임의 확장**

최근 주목받는 Test-Time Scaling (chain-of-thought, best-of-N 등)과 달리, 이 논문은:

$$\text{Wake-time latency 고정} + \text{Sleep-time compute 확장}$$

이라는 새로운 compute 할당 패러다임을 제시합니다. Lin et al. (2025) [35]의 "sleep-time compute"와 상호보완적 관계에 있습니다.

**③ 생물학적 영감의 계산적 구현**

해마 리플레이(hippocampal replay)와 시냅스 공고화(synaptic consolidation)의 계산적 유사체를 언어 모델에서 구현함으로써, 신경과학-AI 상호작용 연구에 새로운 모델을 제공합니다.

**④ Hybrid Architecture 설계 가이드라인**

SSM과 Attention의 하이브리드 설계에 있어, **fast weight 업데이트 규칙의 반복 횟수**가 중요한 설계 변수임을 확립합니다.

### 4.2 향후 연구 시 고려할 점

**① 최적 $N$ 선택 기준 탐구**

현재 논문은 $N = \{2, 3, 4, 6\}$ 등을 실험적으로 탐색하지만, 최적 $N$을 결정하는 이론적 또는 실용적 기준이 부재합니다. Prairie et al. (2026) [46]의 "Parcae: Scaling Laws for Stable Looped Language Models" 연구처럼, sleep depth에 대한 스케일링 법칙 연구가 필요합니다.

**② 학습 안정성 개선**

$N$이 커질수록 역전파가 깊어져 불안정해질 수 있습니다. 고려할 수 있는 접근법:
- **Implicit gradients** (Bai et al., 2019 [4], Deep Equilibrium Models)
- **Truncated BPTT** (McLeish et al., 2025 [39])
- **Progressive training** (낮은 $N$에서 시작하여 점진적으로 증가)

**③ 대규모 사전학습에서의 검증**

현재 실험은 주로 fine-tuning 또는 소규모 scratch training에 한정됩니다. 대규모 사전학습 단계에서 sleep 메커니즘의 효과를 검증하는 것이 필요합니다.

**④ 일반 언어 능력 평가**

MMLU, HellaSwag 등 일반 벤치마크에서의 성능 유지 여부를 확인해야 합니다. Fine-tuning 후 발생할 수 있는 catastrophic forgetting 문제에 대한 대책이 필요합니다.

**⑤ 적응적 Sleep Duration**

모든 컨텍스트 청크에 동일한 $N$을 적용하는 것은 비효율적일 수 있습니다. 청크의 복잡도에 따라 $N$을 동적으로 조정하는 적응적(adaptive) sleep 메커니즘이 필요합니다. Graves (2016) [25]의 Adaptive Computation Time과 결합 가능성이 있습니다.

**⑥ 다양한 SSM Update Rule과의 결합**

현재는 주로 GDN (Gated Delta Network)과 Jet layer를 사용합니다. RWKV, Mamba2 등 다양한 SSM 업데이트 규칙과 sleep 메커니즘의 결합 효과를 탐구할 필요가 있습니다.

**⑦ 멀티모달 및 도메인 특화 적용**

수학 추론 외에도, 긴 문서 이해, 코드 생성, 멀티모달 컨텍스트 처리 등 다양한 도메인에서의 적용 가능성을 탐구해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 긴 컨텍스트 처리 관련 연구 비교

| 연구 | 방법 | 메모리 | 추론 깊이 | Wake Latency | 본 논문과의 관계 |
|------|------|--------|-----------|--------------|----------------|
| **Vaswani et al. (2017)** | Full Attention | $O(T)$ KV cache | 고정 depth | $O(T^2)$ | 기반 아키텍처 |
| **Dao & Gu (2024) [17]** (Mamba2) | SSM (fast weight) | 고정 크기 | 1 pass | $O(1)$ | 본 논문이 확장하는 기반 |
| **Yang et al. (2024) [61]** (GDN) | Delta-rule SSM | 고정 크기 | 1 pass | $O(1)$ | 본 논문의 주요 비교 대상 |
| **Ren et al. (2024) [48]** (Samba) | Sliding-window + SSM hybrid | $O(L)$ + 고정 | 1 pass | $O(L)$ | 본 논문의 베이스라인 |
| **De et al. (2024) [18]** (Griffin) | Gated linear recurrence + local attention | 고정 크기 | 1 pass | $O(L)$ | SSM 하이브리드 비교군 |
| **Dong et al. (2024) [20]** (Hymba) | Hybrid-head | 혼합 | 1 pass | - | SSM 하이브리드 비교군 |
| **Ge et al. (2023) [23]** | 컨텍스트 압축 | 단축 hidden states | 1 pass | - | 방향은 유사, 방법 상이 |
| **Tandon et al. (2025) [55]** | Test-time gradient update (MLP) | 일시적 파라미터 업데이트 | 1 gradient step/chunk | - | 유사한 목표, 1 step 제한 |
| **Lin et al. (2025) [35]** | 오프라인 사전 계산 (질문 예측) | KV cache 재사용 | - | - | 개념적 유사성, 방법 상이 |
| **Eyuboglu et al. (2025) [22]** | Offline self-study (KV cache 압축) | 소규모 KV cache | - | - | 오프라인 학습 공통점 |
| **Geiping et al. (2025) [24]** | Latent reasoning via recurrent depth | - | 가변 (test-time) | 증가 | depth-recurrent 선행연구, wake-time looping |
| **McLeish et al. (2025) [39]** | Retrofitted recurrence (post-training) | - | 가변 (test-time) | 증가 | 본 논문 저자 그룹의 선행연구 |
| **Zhang et al. (2026) [63]** | LoRA adapter (RL) | LoRA 파라미터 | 1 update/chunk | - | 유사한 방향, 1회 업데이트 제한 |
| **본 논문 (2026)** | **Sleep (N offline recurrent passes)** | **Fast weight (SSM)** | **$N$ passes (offline)** | **단일 forward pass** | — |

### 5.2 Depth-Recurrent 모델 계열과의 비교

Geiping et al. (2025) [24]와 McLeish et al. (2025) [39]의 depth-recurrent 접근과 본 논문의 핵심적 차이:

| 특성 | 기존 Depth-Recurrent (Geiping 2025 등) | 본 논문 (LLM Sleep) |
|------|--------------------------------------|---------------------|
| **재귀 시점** | 예측 시간(wake time) | 통합 시간(sleep time) |
| **Gradient 흐름** | 정제된 feature vector 통과 | 정제된 **fast weight** 통과 |
| **예측 지연** | $N$에 비례하여 증가 | **고정 (단일 pass)** |
| **목적** | 예측 정확도 향상 | **메모리 통합 + 추론 능력** |

### 5.3 Test-Time Training/Compute 계열과의 비교

Tandon et al. (2025) [55]의 "End-to-End Test-Time Training for Long Context"와의 차이:

$$\text{Tandon et al.} : \text{cross-entropy loss로 1 gradient step} \times \text{chunk 수}$$

$$\text{본 논문} : \text{learned recurrent forward pass} \times N \text{ (end-to-end 최적화)}$$

핵심 차이는 업데이트 규칙의 유연성에 있습니다. 본 논문의 방법은 특정 손실 함수에 구애받지 않는 학습된 규칙을 사용합니다.

### 5.4 Sleep-Time Compute (Lin et al., 2025) [35]와의 관계

Lin et al. (2025)의 "Sleep-Time Compute"는 LLM이 예상 질문을 생성하고 필요한 계산을 사전 수행하는 방법인 반면, 본 논문은 fast weight 자체를 반복적으로 정제하는 방법입니다. 두 방법은 상호보완적으로 결합될 수 있습니다.

---

## 참고 자료

**본 논문 (주 출처):**
- Lee, S., McLeish, S., Goldstein, T., & Fanti, G. (2026). *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference.* arXiv:2605.26099v2.

**논문 내 인용 핵심 참고문헌:**
- [1] Allen-Zhu & Li (2025). *Physics of Language Models: Part 4.1.* NeurIPS 2025.
- [2] Arora et al. (2024). *Simple linear attention language models balance the recall-throughput tradeoff.* arXiv:2402.18668.
- [4] Bai, Kolter & Koltun (2019). *Deep equilibrium models.* NeurIPS 32.
- [17] Dao & Gu (2024). *Transformers are SSMs.* arXiv:2405.21060.
- [18] De et al. (2024). *Griffin: Mixing gated linear recurrences with local attention.* arXiv:2402.19427.
- [19] Dehghani et al. (2018). *Universal Transformers.* arXiv:1807.03819.
- [20] Dong et al. (2024). *Hymba: A hybrid-head architecture.* arXiv:2411.13676.
- [22] Eyuboglu et al. (2025). *Cartridges: Lightweight and general-purpose long context representations via self-study.* arXiv:2506.06266.
- [24] Geiping et al. (2025). *Scaling up test-time compute with latent reasoning.* NeurIPS 2025.
- [25] Graves (2016). *Adaptive computation time for recurrent neural networks.* arXiv:1603.08983.
- [26] Gu et al. (2025). *Jet-Nemotron.* arXiv:2508.15884.
- [32] Jelassi et al. (2024). *Repeat after me: Transformers are better than SSMs at copying.* arXiv:2402.01032.
- [33] Kabra et al. (2026). *Learning from synthetic data improves multi-hop reasoning.* ICLR 2026.
- [35] Lin et al. (2025). *Sleep-time compute: Beyond inference scaling at test-time.* arXiv:2504.13171.
- [37] Liu et al. (2025). *The serial scaling hypothesis.* arXiv:2507.12549.
- [39] McLeish et al. (2025). *Teaching pretrained language models to think deeper with retrofitted recurrence.* arXiv:2511.07384.
- [40] Merrill & Sabharwal (2025). *A little depth goes a long way.* arXiv:2503.03961.
- [46] Prairie et al. (2026). *Parcae: Scaling laws for stable looped language models.* arXiv:2604.12946.
- [48] Ren et al. (2024). *Samba: Simple hybrid state space models.* arXiv:2406.07522.
- [51] Schwethelm et al. (2026). *How much is one recurrence worth?* arXiv:2604.21106.
- [55] Tandon et al. (2025). *End-to-end test-time training for long context.* arXiv:2512.23675.
- [58] Vaswani et al. (2017). *Attention is all you need.* NeurIPS 2017.
- [61] Yang et al. (2024). *Gated delta networks: Improving Mamba2 with delta rule.* arXiv:2412.06464.
- [64] Zhou et al. (2025). *GSM-Infinite.* ICML 2025 Workshop.
- [65] Zhu et al. (2025). *Scaling latent reasoning via looped language models.* arXiv:2510.25741.
