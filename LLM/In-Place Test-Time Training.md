# In-Place Test-Time Training (In-Place TTT)
---

## 참고 자료

- **주요 논문**: Feng, G., Luo, S., Hua, K., Zhang, G., He, D., Huang, W., & Cai, T. (2026). *In-Place Test-Time Training*. arXiv:2604.06169v1. https://arxiv.org/abs/2604.06169
- **관련 논문들**: 논문 내 참고문헌 [2], [3], [47], [48], [67] 등 (상세 목록은 각 섹션에 명시)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 LLM의 "train then deploy" 패러다임은 추론 시점에 모델 가중치를 동적으로 갱신하지 못한다는 근본적 한계를 가집니다. Test-Time Training(TTT)이 이를 해결할 수 있는 대안이지만, 기존 LLM 생태계에 적용하기 위해서는 세 가지 장벽이 존재합니다:

| 장벽 | 내용 |
|------|------|
| **(i) 아키텍처 비호환성** | 기존 TTT는 전문 레이어가 필요해 처음부터 재학습 필요 |
| **(ii) 계산 비효율성** | 토큰별 순차 업데이트로 병렬처리 불가 |
| **(iii) 목적함수 불일치** | 재구성(reconstruction) 목적은 LM의 NTP 목표와 불일치 |

**In-Place TTT**는 이 세 장벽을 동시에 해결하는 프레임워크입니다.

### 주요 기여

1. **"Drop-in" 설계**: 기존 MLP 블록의 $\mathbf{W}_{\text{down}}$을 fast weights로 재활용 → 재학습 없이 적용 가능
2. **청크 단위 업데이트**: 병렬 처리 친화적인 chunk-wise update mechanism 설계
3. **LM-Aligned Objective**: NTP(Next-Token Prediction)에 명시적으로 정렬된 이론적 근거 있는 목적함수 제안
4. **Context Parallelism 호환**: Parallel Scan 알고리즘으로 확장 가능한 구현 실현
5. **광범위한 실험 검증**: Qwen3-4B, LLaMA-3.1-8B, Qwen3-14B에서 일관된 성능 향상 확인

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 정의**: LLM은 배포 후 가중치가 고정되어, 연속적으로 들어오는 스트리밍 입력에 적응하지 못합니다. In-context learning은 문맥 창 길이에 제한되고, 기존 TTT는 LLM 생태계에 통합하기 어렵습니다.

**구체적 세 가지 장벽**:

- **장벽 (i)**: TTT는 독립적인 순환 레이어 형태로, 사전학습 체크포인트를 활용할 수 없음 (Sun et al., 2024 [48]; Zhang et al., 2025 [67])
- **장벽 (ii)**: 토큰별 순차 업데이트는 GPU/TPU의 병렬성을 저해
- **장벽 (iii)**: 일반적인 재구성 목적함수($\mathcal{L}(f_W(k), v)$, $v \leftarrow x_i$)는 NTP와 직접적으로 연관 없음

---

### 2.2 제안 방법 및 수식

#### (A) 기본 TTT 메커니즘

입력 시퀀스 $\mathbf{x} = [x_1, x_2, \ldots, x_N]$에 대해 fast weights $\mathbf{W}$를 다음과 같이 업데이트합니다:

$$\mathbf{W}_i \leftarrow \mathbf{W}_{i-1} - \eta \nabla_{\mathbf{W}} \mathcal{L}\left(f_{\mathbf{W}_{i-1}}(k_i),\, v_i\right)$$

출력: $o_i = f_{\mathbf{W}_i}(q_i)$

---

#### (B) MLP 블록 재활용 (핵심 설계)

Gated MLP의 출력은 다음과 같이 정의됩니다:

```math
\mathbf{O} = \left(\phi(\mathbf{H}\mathbf{W}_{\text{gate}}^\top) \odot (\mathbf{H}\mathbf{W}_{\text{up}}^\top)\right)\mathbf{W}_{\text{down}}^\top
```

- $\mathbf{W}\_{\text{up}}, \mathbf{W}_{\text{gate}}$: **고정** (slow weights)
- $\mathbf{W}_{\text{down}}$: **fast weights**로 재활용 → in-place 업데이트

중간 활성화:

$$\mathbf{Z} = \phi(\mathbf{H}\mathbf{W}_{\text{gate}}^\top) \odot (\mathbf{H}\mathbf{W}_{\text{up}}^\top) \in \mathbb{R}^{n \times d_{\text{ff}}}$$

---

#### (C) 청크 단위 업데이트

시퀀스를 $k$개의 청크($C$ 크기)로 분할하여 각 청크 $i$에 대해:

**1. Apply Operation:**

$$\mathbf{O}_{[i]} = \mathbf{Z}_{[i]} \left(\mathbf{W}_{\text{down}}^{(i)}\right)^\top$$

**2. Update Operation:**

$$\mathbf{W}_{\text{down}}^{(i+1)} = \mathbf{W}_{\text{down}}^{(i)} - \eta \nabla_{\mathbf{W}} \mathcal{L}\left(\mathbf{Z}_{[i]}\left(\mathbf{W}_{\text{down}}^{(i)}\right)^\top,\, \mathbf{V}_{[i]}\right)$$

---

#### (D) LM-Aligned Objective (핵심 기여)

NTP 목표에 맞춘 타겟 $\hat{\mathbf{V}}$를 다음과 같이 정의합니다:

$$\hat{\mathbf{V}} = \text{Conv1D}(\mathbf{X}_0)\mathbf{W}_{\text{target}}$$

- $\mathbf{X}\_0 \in \mathbb{R}^{n \times d_{\text{model}}}$: 토큰 임베딩
- $\text{Conv1D}(\cdot)$: 미래 토큰 정보를 포함 (causal padding 적용)
- $\mathbf{W}\_{\text{target}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$: 학습 가능한 투영 행렬

손실함수: $\mathcal{L}(\cdot, \cdot) = -\langle \cdot, \cdot \rangle_F$ (음의 Frobenius inner product)

이 경우 **fast weights 업데이트 규칙**은 단순화됩니다:

$$\boxed{\mathbf{W}_{\text{down}}^{(i)} = \mathbf{W}_{\text{down}}^{(i-1)} + \eta \hat{\mathbf{V}}_{[i]}^\top \mathbf{Z}_{[i]}}$$

---

#### (E) Context Parallelism 구현

업데이트의 결합성(associativity)을 활용한 3단계 병렬화:

**Stage 1** (병렬): 모든 청크에 대해 업데이트 델타 계산:

$$\Delta \mathbf{W}^{(i)}_{\text{down}} = (\hat{\mathbf{V}}_{[i]})^\top \mathbf{Z}_{[i]}$$

**Stage 2** (Prefix Sum): 누적 업데이트 계산:

$$\Delta \mathbf{S}_i = \sum_{j=1}^{i-1} \Delta \mathbf{W}^{(j)}$$

**Stage 3** (병렬): 유효 fast weights 및 출력 계산:

$$\mathbf{W}_{\text{down}}^{(i-1)} = \mathbf{W}_{\text{down}}^{(0)} + \eta \Delta \mathbf{S}_i, \quad \mathbf{O}_{[i]} = \mathbf{Z}_{[i]}\left(\mathbf{W}_{\text{down}}^{(i-1)}\right)^\top$$

---

### 2.3 모델 구조

```
입력 토큰 시퀀스
       ↓
[Attention Block] (변경 없음, 고정)
       ↓
[Gated Linear Layer: W_up, W_gate] (고정, slow weights)
       ↓
Z = φ(H·W_gate^T) ⊙ (H·W_up^T)
       ↓
청크 분할 (C = 512 or 1024)
       ↓
┌─────────────────────────────┐
│  For each chunk i:          │
│  1. Apply: O[i] = Z[i]·W_down^(i)^T  │
│  2. V[i] = Conv1D(X0)·W_target       │
│  3. Update: W_down^(i+1) += η·V[i]^T·Z[i] │
└─────────────────────────────┘
       ↓
출력 O (풍부해진 컨텍스트 정보 포함)
```

**초기화 전략**: continual training 시 Conv1D는 zero 초기화, $\mathbf{W}_{\text{target}}$은 희소 대각행렬로 초기화 → 초기 fast weight 업데이트 $\approx 0$이 되어 사전학습 능력 보존

---

### 2.4 성능 향상

#### RULER 벤치마크 (Qwen3-4B-Base 기반)

| 모델 | 4k | 8k | 16k | 32k | 64k | 128k | 256k |
|------|----|----|-----|-----|-----|------|------|
| Baseline | **96.6** | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| **In-Place TTT** | 96.1 | **95.6** | **92.7** | **89.3** | **78.7** | **77.0** | **43.9** |

- 128k에서 +2.2%, 256k 외삽(Extrapolation)에서 +2.2% 향상

#### 다양한 모델 확장성

| 베이스 모델 | 메서드 | 64k | 64k+YaRN |
|------------|--------|-----|----------|
| LLaMA-3.1-8B | Baseline | 81.6 | - |
| LLaMA-3.1-8B | **In-Place TTT** | **83.7** | - |
| Qwen3-14B | Baseline | 67.9 | 81.3 |
| Qwen3-14B | **In-Place TTT** | **70.6** | **82.5** |

#### 처음부터 학습 (500M/1.5B)

Sliding Window Perplexity 기준으로 **모든 경쟁 기법** (SWA, GLA, DeltaNet, LaCT)을 **일관되게 능가**

#### 4B 모델 상식 추론 + 장문 평가

| 구성 | HellaSwag | ARC-C | MMLU | RULER-16k |
|------|-----------|-------|------|-----------|
| Full Attn. Baseline | 55.67 | 33.19 | 36.43 | 6.58 |
| **In-Place TTT (Full)** | **55.85** | 32.34 | **37.42** | **19.99** |
| SWA Baseline | 54.92 | 32.85 | 36.06 | 5.07 |
| **In-Place TTT (SWA)** | 55.24 | **33.70** | 36.48 | **26.80** |

---

### 2.5 한계

1. **클리핑 메커니즘 의존성**: 장문 추론 시 Frobenius norm 클리핑 ($\tau = 10^{-5}$)이 필요 → 이론적으로 완전한 해결책이 아님
2. **Loss function 및 optimizer 미탐색**: 저자들 스스로 다양한 loss/optimizer 조합을 미래 연구로 남겨둠
3. **효율성 지표 불완전**: 처리량과 메모리 오버헤드가 negligible하다고 주장하지만, 128k 이상 매우 장문에서의 상세한 오버헤드 분석은 제한적
4. **Short context 성능 trade-off**: 일부 단문(4k) 설정에서는 baseline이 미세하게 높은 경우 존재
5. **TTT 활성화 레이어 비율**: 6번째 레이어마다 적용 → 최적 레이어 선택 전략이 아직 휴리스틱

---

## 3. 일반화 성능 향상 가능성 (심층 분석)

### 3.1 이론적 근거: Theorem 1 분석

**Induction Head 설정** 하에서 LM-Aligned Target과 Reconstruction Target의 로짓 변화를 비교합니다.

**설정**: 시퀀스에서 위치 $t^\*$에 key-value 쌍 $(k^\*, v^\*)$이 등장하고, 위치 $n > t^\*$에서 쿼리 $x_n = k^\*$가 재등장. 모델은 $x_{n+1} = v^*$를 예측해야 함.

**Fast weights 변화량**: $\Delta \mathbf{W}\_{\text{down}} = \eta \sum_{t \in \text{prior}} \mathbf{v}_t \mathbf{z}_t^\top$

로짓 변화: $\Delta \ell_n[w] = \mathbf{e}\_w^\top (\Delta \mathbf{W}\_{\text{down}}) \mathbf{z}\_n = \lambda_{lr} \sum_{t \in \text{prior}} (\mathbf{e}_w^\top \mathbf{v}_t)(\mathbf{z}_t^\top \mathbf{z}_n)$

**Theorem 1 (LM-Aligned vs Reconstruction)**:

*LM-Aligned Target* ($\mathbf{v}\_{t^\*} = \mathbf{e}\_{x_{t^*+1}}$):

$$\mathbb{E}[\Delta \ell_n[v^*]] \geq \lambda_{lr} \cdot c_{\text{norm}}^2 \cdot c_{\text{align}} \quad \text{(올바른 토큰 로짓 증가)}$$

$$|\mathbb{E}[\Delta \ell_n[w]]| \leq \lambda_{lr} \cdot \epsilon \cdot c_{\text{align}}, \quad \forall w \neq v^* \quad \text{(다른 토큰 로짓 거의 불변)}$$

*Reconstruction Target* ($\mathbf{v}\_{t^\*} = \mathbf{e}\_{x_{t^*}}$):

$$|\mathbb{E}[\Delta \ell_n[v^*]]| \leq \lambda_{lr} \cdot \epsilon \cdot c_{\text{align}} \quad \text{(올바른 토큰에 효과 없음)}$$

**핵심 의미**: LM-Aligned Target은 fast weights가 **미래 예측에 유용한 정보**를 압축하도록 유도하는 반면, Reconstruction Target은 단순히 현재 토큰을 기억하도록 유도하여 NTP에 직접적 도움을 주지 못합니다.

---

### 3.2 일반화 성능 향상의 다층적 메커니즘

#### (a) 장문 컨텍스트 일반화

| 검증 방식 | 결과 |
|-----------|------|
| 128k 훈련 → 256k 외삽 | In-Place TTT: 43.9 vs Baseline: 41.7 |
| 64k 이상 장문에서 격차 증가 | 컨텍스트가 길수록 TTT 효과 증폭 |

Fast weights가 컨텍스트 정보를 동적으로 압축하기 때문에, 정적인 attention이 처리하기 어려운 초장문에서의 일반화 능력이 향상됩니다.

#### (b) 도메인 간 일반화

모델 패밀리 (Qwen3, LLaMA)와 스케일 (4B~14B)에 걸쳐 일관된 향상이 확인됩니다:

$$\Delta_{\text{RULER-64k}} \approx +2.1 \sim +2.7\%\text{ (모든 모델 패밀리)}$$

#### (c) Induction Head 강화를 통한 ICL 일반화

Theorem 1에서 보이듯, LM-Aligned Objective는 induction head 메커니즘을 명시적으로 강화합니다. Induction head는 **In-Context Learning(ICL)의 핵심 메커니즘** (Olsson et al., 2022 [38])이므로, 이 강화는 분포 외(out-of-distribution) 패턴에 대한 일반화로 이어집니다.

#### (d) Multi-Token Prediction과의 연계

논문에서 언급된 바와 같이, Conv1D를 통한 localized future token combination은 **Multi-Token Prediction** (DeepSeek-V3, [37])의 원리와 일치합니다:

$$\hat{\mathbf{V}} = \text{Conv1D}(\mathbf{X}_0)\mathbf{W}_{\text{target}}$$

이는 단순한 next-token보다 더 풍부한 예측 신호를 압축하여 일반화 성능을 향상시킵니다.

#### (e) State Size와 성능의 스케일링 법칙

Ablation 결과:

$$\text{성능}(4\times \text{state}) > \text{성능}(1\times \text{state}) > \text{성능}(0.5\times \text{state})$$

Fast weights 크기가 클수록 더 많은 컨텍스트 정보를 압축 가능하며, 이는 MLP의 key-value memory 특성 [22]을 일반화 목적으로 활용하는 것과 일치합니다.

#### (f) YaRN과의 상보성

Position extrapolation 기법(YaRN [42])과 결합 시에도 독립적인 성능 향상이 유지됩니다 (+1.2% at 64k+YaRN for Qwen3-14B). 이는 In-Place TTT의 일반화 효과가 positional encoding 확장과 직교(orthogonal)함을 의미합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 TTT 계열 연구

| 논문 | 연도 | 핵심 아이디어 | In-Place TTT와 차이점 |
|------|------|--------------|----------------------|
| Sun et al. (TTT, [47]) | 2020 | Self-supervised TTT for distribution shift (CV) | 언어모델 전용 아님, 순차 업데이트 |
| Wang et al. (TENT, [53]) | 2021 | Entropy minimization 기반 test-time adaptation | LLM 비적용, 구조 변경 필요 |
| Sun et al. (TTT-LM, [48]) | 2024 | RNN 기반 expressive hidden state | 새 레이어 필요, NTP 불일치 |
| Behrouz et al. (Titans, [3]) | 2024 | Test-time memorization, chunk-wise | 독립 레이어 설계, attention 대체 |
| Zhang et al. (LaCT, [67]) | 2025 | Large Chunk TTT | SWA 위에서만 동작, reconstruction 목적 |
| **In-Place TTT** | **2026** | **MLP in-place 재활용 + NTP-aligned** | **Drop-in, 이론적 근거, CP 호환** |

### 4.2 효율적 장문 처리 연구

| 논문 | 연도 | 방법 | In-Place TTT와 관계 |
|------|------|------|---------------------|
| Beltagy et al. (Longformer, [6]) | 2020 | Sliding Window Attention | 상보적 (SWA 위에서도 동작) |
| Katharopoulos et al. (Linear Attn, [33]) | 2020 | Linear attention | 상보적, MLP 블록 공유 |
| Yang et al. (GLA, [60]) | 2023/24 | Gated Linear Attention | 비교 기준선으로 사용 |
| Yang et al. (DeltaNet, [61]) | 2024 | Delta rule 병렬화 | 비교 기준선으로 사용 |
| Peng et al. (YaRN, [42]) | 2023 | RoPE 확장 | 직교적 결합 가능 |

### 4.3 메모리 증강 연구

| 논문 | 연도 | 방법 | In-Place TTT와 관계 |
|------|------|------|---------------------|
| Khandelwal et al. (kNN-LM, [34]) | 2020 | Nearest neighbor 언어모델 | 외부 메모리 vs 내부 fast weights |
| Lewis et al. (RAG, [35]) | 2020 | Retrieval-augmented generation | 외부 검색 vs 내재화 |
| Wang et al. (MemoryLLM, [55]) | 2024 | Self-updatable LLM | 구조 수정 필요 |
| Behrouz et al. (It's All Connected, [4]) | 2025 | TTT-attention bias 연결 분석 | In-Place TTT의 이론적 배경 보완 |

### 4.4 비교 분석 요약

```
┌────────────────────────────────────────────────────────┐
│          방법론 비교 매트릭스                           │
├─────────────┬──────┬──────┬──────┬──────┬─────────────┤
│ 방법        │Drop- │Chunk-│NTP-  │CP    │LLM 사전학습│
│             │in    │wise  │Align │호환  │보존        │
├─────────────┼──────┼──────┼──────┼──────┼─────────────┤
│ TTT-LM [48] │ ✗    │ ✗    │ ✗    │ ✗    │ ✗          │
│ Titans [3]  │ ✗    │ ✓    │ ✗    │ △    │ ✗          │
│ LaCT [67]   │ ✗    │ ✓    │ ✗    │ △    │ ✗          │
│ In-Place TTT│ ✓    │ ✓    │ ✓    │ ✓    │ ✓          │
└─────────────┴──────┴──────┴──────┴──────┴─────────────┘
```

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### (A) 패러다임 전환: "Continual Learning in LLMs"

In-Place TTT는 LLM 추론을 **동적 학습 과정**으로 재개념화합니다. 이는 다음 연구 방향을 촉진합니다:

- **On-the-fly domain adaptation**: 특정 도메인 문서를 읽으면서 실시간 적응
- **Streaming data 처리**: 끊임없이 업데이트되는 정보 스트림에의 연속 적응
- **Personalized LLM**: 사용자 상호작용 이력을 fast weights에 내재화

#### (B) MLP 블록의 재해석

Geva et al. (2020, [22])의 "Transformer FFN은 key-value 메모리"라는 관점을 발전시켜, **slow weights(일반 지식) + fast weights(문맥 지식)** 이중 메모리 구조를 확립합니다.

#### (C) 이론적 기여: NTP-Aligned Objective 프레임워크

Theorem 1은 **어떤 TTT 목적함수가 언어모델링에 적합한가**에 대한 최초의 이론적 분석 중 하나입니다. 이 프레임워크는 향후 다양한 모달리티(코드, 수학 등)에 특화된 TTT 목적함수 설계의 기준을 제공합니다.

#### (D) 효율적 아키텍처 연구에의 영향

In-Place TTT가 SWA, GLA, Full Attention 등 다양한 백본과 결합 가능함을 보임으로써, **하이브리드 아키텍처** 연구(Attention + Linear Recurrence + TTT)의 새로운 방향을 제시합니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (A) 목적함수 및 최적화 전략 탐색

저자들이 명시적으로 미래 연구로 남겨둔 영역:

$$\text{Future Work: } \mathcal{L}_{\text{TTT}} \in \{\text{MSE, Cosine, Contrastive, ...}\}, \quad \text{Optimizer} \in \{\text{SGD, Adam, ...}\}$$

특히 **Momentum을 활용한 fast weight 업데이트**나 **Second-order 최적화**가 성능을 향상시킬 가능성이 있습니다.

#### (B) 레이어별 TTT 활성화 전략

현재 6번째 레이어마다 적용하는 휴리스틱에서 벗어나, **레이어 중요도에 따른 선택적 TTT 활성화**를 탐구해야 합니다:

$$\text{Selective TTT: } \mathbf{W}_{\text{down}}^{(l)} \text{ 업데이트 여부를 학습으로 결정}$$

#### (C) Forgetting 메커니즘

문서 경계에서 fast weights를 초기화하는 현재 방식은 단순하지만, **선택적 망각(selective forgetting)** 메커니즘을 통해 장기적으로 유용한 정보를 유지하는 방향이 필요합니다:

$$\mathbf{W}_{\text{down}}^{(i+1)} = \lambda \mathbf{W}_{\text{down}}^{(i)} + \eta \hat{\mathbf{V}}_{[i]}^\top \mathbf{Z}_{[i]}, \quad \lambda \in (0, 1)$$

이는 **Continual Learning의 catastrophic forgetting** 문제와 직결됩니다.

#### (D) 다중 모달리티 확장

논문에서는 언어 모달리티만 다루지만, **코드, 수학, 멀티모달(이미지+텍스트)** 에서의 TTT 목적함수 설계가 필요합니다. 특히 코드와 수학은 구조적 패턴이 강해 특화된 목적함수가 효과적일 수 있습니다.

#### (E) 이론적 보완

Theorem 1은 다음 가정에 의존합니다:

1. 임베딩 간 근사 직교성: $|\mathbf{e}\_{w_i}^\top \mathbf{e}_{w_j}| \leq \epsilon$
2. Key-Query 정렬: $\mathbb{E}[\mathbf{z}\_{t^\*}^\top \mathbf{z}\_n] = c_{\text{align}} > 0$

실제 대규모 LLM에서 이 가정의 성립 정도를 경험적으로 검증하고, **더 약한 가정** 하에서의 이론적 분석이 필요합니다.

#### (F) 클리핑 없는 안정적 학습

현재 $\|\Delta \mathbf{W}_{\text{down}}^{(i)}\|_F > \tau$ 시 클리핑이 필요한 불안정성을 근본적으로 해결하기 위한 **정규화 방법론** 연구가 필요합니다 (예: spectral normalization, gradient clipping alternatives).

#### (G) 사전학습 단계에서의 TTT 통합

현재는 continual training 또는 from-scratch training이지만, **사전학습 단계부터 In-Place TTT를 포함한 훈련**이 더 큰 효과를 가져올 수 있는지 검토가 필요합니다.

#### (H) 다운스트림 태스크에서의 검증

현재 실험이 언어모델링(perplexity, RULER)에 집중되어 있어, **복잡한 추론 태스크** (수학 증명, 코드 생성, 다단계 추론)에서의 효과 검증이 필요합니다.

---

## 결론

In-Place TTT는 LLM의 정적 추론 패러다임을 **동적 적응 패러다임**으로 전환하는 실용적이고 이론적으로 근거 있는 프레임워크입니다. MLP 블록 재활용이라는 단순하지만 강력한 통찰, NTP-aligned objective의 이론적 우월성 증명, 그리고 Context Parallelism과의 호환성이 결합되어 기존 TTT의 세 가지 핵심 장벽을 효과적으로 극복합니다. 특히 일반화 성능 측면에서 장문 컨텍스트, 다양한 모델 패밀리, 분포 외 시나리오에서 일관된 향상을 보임으로써, 향후 LLM의 continual learning 연구에 중요한 기준점을 제시합니다.
