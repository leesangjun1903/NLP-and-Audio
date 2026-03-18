# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

---

## 1. 핵심 주장과 주요 기여 요약

이 논문은 Transformer 기반 대규모 언어 모델(LLM)의 **디코더 추론 속도 병목 문제**를 해결하기 위해 두 가지 핵심 기여를 제시한다:

1. **Uptraining 기법**: 기존 Multi-Head Attention(MHA) 체크포인트를 Multi-Query Attention(MQA) 또는 Grouped-Query Attention(GQA) 구조로 변환한 뒤, 원래 사전 학습 계산량의 **약 5%만 추가 학습**하여 효율적으로 변환하는 방법론을 제안.
2. **Grouped-Query Attention(GQA)**: MHA와 MQA의 **중간 형태(interpolation)**로서, 쿼리 헤드를 $G$개의 그룹으로 나누어 각 그룹이 하나의 Key-Value 헤드를 공유하는 새로운 어텐션 메커니즘을 제안. **MHA에 근접한 품질**을 유지하면서 **MQA에 근접한 추론 속도**를 달성함을 실험적으로 입증.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

자기회귀(autoregressive) 디코더 추론 시, 매 디코딩 스텝마다 디코더 가중치와 **모든 어텐션 Key-Value(KV) 캐시를 메모리에서 로드**해야 하는 **메모리 대역폭(memory bandwidth) 오버헤드**가 심각한 병목이 된다 (Shazeer, 2019; Pope et al., 2022).

기존 MQA(Shazeer, 2019)는 KV 헤드를 단 하나로 줄여 이 문제를 크게 완화하지만, 두 가지 한계가 있다:

- **품질 저하(quality degradation)** 및 **학습 불안정성(training instability)** 발생
- 기존에 MHA로 학습된 수많은 공개 모델(T5, LLaMA 등)을 처음부터 다시 학습하기 어려움

### 2.2 제안하는 방법

#### (A) Multi-Head Attention (MHA) — 기준선

표준 MHA에서 $H$개의 쿼리, 키, 밸류 헤드가 각각 존재한다. 어텐션 연산은 다음과 같다:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

여기서 $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$이며, 각 헤드 $i$에 대해:

$$Q_i = XW_i^Q, \quad K_i = XW_i^K, \quad V_i = XW_i^V$$

MHA에서는 $H$개의 독립적인 $W_i^K$, $W_i^V$ 프로젝션이 존재하므로, KV 캐시 크기가 $H$에 비례하여 메모리 대역폭 부담이 크다.

#### (B) Multi-Query Attention (MQA)

MQA에서는 **모든 쿼리 헤드가 단일 Key 헤드와 단일 Value 헤드를 공유**한다:

$$K = XW^K, \quad V = XW^V$$

$$\text{head}_i = \text{Attention}(Q_i, K, V)$$

이를 통해 KV 캐시 크기가 $H$배 감소한다. 그러나 모델 용량이 줄어들어 품질 저하가 발생할 수 있다.

#### (C) Grouped-Query Attention (GQA) — 본 논문의 핵심 제안

GQA는 $H$개의 쿼리 헤드를 $G$개의 **그룹**으로 나누고, 각 그룹 내에서 하나의 Key 헤드와 Value 헤드를 공유한다:

$$\text{GQA-}G: \quad \text{그룹 } g \text{에 속하는 쿼리 헤드들이 } K_g, V_g \text{를 공유}$$

$$K_g = XW_g^K, \quad V_g = XW_g^V, \quad g = 1, 2, \ldots, G$$

$$\text{head}_i = \text{Attention}(Q_i, K_{g(i)}, V_{g(i)})$$

여기서 $g(i)$는 쿼리 헤드 $i$가 속한 그룹의 인덱스이다.

**특수 케이스:**
- $G = 1$: **GQA-1 = MQA** (단일 KV 헤드)
- $G = H$: **GQA-H = MHA** (헤드 수만큼의 KV 헤드)
- $1 < G < H$: MQA와 MHA 사이의 보간(interpolation)

KV 캐시 크기는 $G$에 비례하므로, MHA 대비 $\frac{H}{G}$배 감소한다.

#### (D) Uptraining 절차

1. **체크포인트 변환**: MHA 체크포인트의 Key/Value 프로젝션 행렬을 그룹별로 **평균 풀링(mean pooling)**하여 GQA 또는 MQA 구조로 변환:

$$W_g^K = \frac{1}{|S_g|}\sum_{i \in S_g} W_i^K, \quad W_g^V = \frac{1}{|S_g|}\sum_{i \in S_g} W_i^V$$

여기서 $S_g$는 그룹 $g$에 속하는 원래 헤드 인덱스 집합이다.

2. **추가 사전 학습**: 변환된 체크포인트를 원래 사전 학습 데이터셋과 동일한 설정으로 **원래 학습 스텝의 $\alpha$ 비율만큼** 추가 학습 ($\alpha = 0.05$, 즉 5%).

### 2.3 모델 구조

- 기반 아키텍처: **T5.1.1** (encoder-decoder 구조)
- GQA/MQA는 **디코더 self-attention**과 **cross-attention**에만 적용, **인코더 self-attention에는 미적용** (인코더는 병렬 처리되므로 메모리 대역폭이 주요 병목이 아님)
- 실험 모델: T5 Large, T5 XXL, 그리고 uptrained MQA-XXL, GQA-8-XXL

### 2.4 성능 향상

| 모델 | 추론 시간 ($T_{\text{infer}}$, s) | 평균 성능 |
|------|:---:|:---:|
| MHA-Large | 0.37 | 46.0 |
| MHA-XXL | 1.51 | 47.2 |
| MQA-XXL (uptrained 5%) | 0.24 | 46.6 |
| **GQA-8-XXL** (uptrained 5%) | **0.28** | **47.1** |

핵심 결과:
- **GQA-8-XXL**은 MHA-XXL 대비 추론 속도가 **약 5.4배 빠르면서** 평균 성능이 47.1로 MHA-XXL(47.2)에 **매우 근접**
- MQA-XXL 대비 속도는 유사하면서(0.28 vs 0.24) 품질이 유의미하게 향상(47.1 vs 46.6)
- Uptraining에 필요한 계산량은 원래 사전 학습의 **단 5%** (약 600 TPUv3 chip-days)

**Ablation 결과:**
- **체크포인트 변환 방법**: Mean pooling > 단일 헤드 선택 > 랜덤 초기화 (사전 학습 정보 보존 정도에 비례)
- **Uptraining 비율**: 5%에서 대부분의 성능 회복, 10%에서 수확 체감
- **GQA 그룹 수**: 1(MQA)에서 8까지 증가 시 추론 시간 증가가 미미하고, 8 이상에서 비용이 급격히 증가 → **8 그룹이 최적의 트레이드오프**

### 2.5 한계

1. **평가 지표의 한계**: 요약 태스크에서 ROUGE 점수를 사용하나, 이는 불완전한 평가 지표
2. **Scratch 학습과의 비교 부재**: GQA 모델을 처음부터 학습한 경우와의 비교 실험이 없음
3. **Encoder-decoder 모델에 한정**: Decoder-only 모델(GPT 계열)에 대한 실험이 없음 (저자들은 decoder-only 모델에서 GQA가 더 큰 이점을 가질 것으로 예상)
4. **MQA의 학습 불안정성**: 특히 긴 입력 태스크에서 MQA는 fine-tuning 시 발산하는 문제가 관찰됨 (Appendix A)

---

## 3. 모델의 일반화 성능 향상 가능성

GQA가 일반화 성능을 향상시키는 메커니즘과 근거를 다각도로 분석한다:

### 3.1 용량 보존을 통한 표현력 유지

MQA는 KV 헤드를 1개로 극단적으로 줄이므로 모델 용량(capacity)이 크게 감소한다. 특히 모델이 커질수록 헤드 수 $H$가 증가하므로, MQA의 용량 감소는 **모델 크기에 비례하여 더 공격적**이 된다:

$$\text{MQA KV cache 감소 비율} = H \quad (\text{대형 모델일수록 } H \text{가 크므로 더 급격한 감소})$$

GQA는 $G$개의 KV 헤드를 유지함으로써, 모델 크기 증가에 따른 **비례적 감소(proportional decrease)**를 유지할 수 있다:

$$\text{GQA KV cache 크기} \propto G \cdot d_k, \quad \text{감소 비율} = \frac{H}{G}$$

이는 다양한 태스크에 대한 **일반화 성능 유지**에 직접적으로 기여한다.

### 3.2 학습 안정성과 일반화

부록 A에서 보고된 바와 같이, MQA로 처음부터 학습한 T5-Large 모델은:
- 사전 학습 중 **빈번한 loss spike**가 발생
- 긴 입력 태스크 fine-tuning 시 **즉시 발산(diverge)**

반면 **uptrained GQA 모델은 안정적**으로 학습되었다. 학습 안정성은 다양한 downstream 태스크에 대한 일반화 성능의 전제 조건이므로, GQA는 MQA 대비 **더 신뢰할 수 있는 일반화 성능**을 제공한다.

### 3.3 다양한 태스크에서의 일관된 성능

Table 1에서 GQA-8-XXL은 **7개 태스크(요약, 번역, 질의응답)** 모두에서 MQA-XXL을 일관되게 상회하며, MHA-XXL에 근접한 성능을 보인다:

- 요약(CNN, arXiv, PubMed, MediaSum, MultiNews): ROUGE-1 기준 일관된 우위
- 번역(WMT): BLEU 28.4 (MHA-XXL 28.4와 동일)
- 질의응답(TriviaQA): F1 81.6 (MHA-XXL 81.9에 근접)

이러한 **태스크 간 일관성**은 GQA의 일반화 능력을 강력히 시사한다.

### 3.4 대형 모델에서의 스케일링 특성

저자들은 대형 모델에서 GQA가 특히 유리한 트레이드오프를 제공할 것으로 분석한다:
- KV 캐시 크기는 $d_{\text{model}}$에 비례하여 선형 증가
- 모델 FLOPs와 파라미터 수는 $d_{\text{model}}^2$에 비례하여 증가
- 따라서 대형 모델일수록 **어텐션으로 인한 메모리 대역폭 오버헤드의 상대적 비중이 감소**
- 대형 모델의 표준 샤딩(sharding)에서 MQA는 단일 KV 헤드를 파티션 수만큼 복제하여 **낭비**가 발생하나, GQA는 그룹 수를 파티션 수에 맞춤으로써 이 낭비를 제거

### 3.5 Uptraining을 통한 사전 학습 지식 보존

Mean pooling 기반 체크포인트 변환은 기존 MHA 모델의 학습된 표현을 **최대한 보존**한다:

$$W_{\text{GQA},g}^K = \frac{1}{|S_g|}\sum_{i \in S_g} W_{\text{MHA},i}^K$$

이는 랜덤 초기화나 단일 헤드 선택보다 우수하며, 사전 학습에서 습득한 일반화 지식을 GQA 구조로 효과적으로 전이할 수 있게 한다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구 영향

1. **산업 표준으로의 채택**: GQA는 이미 Meta의 **LLaMA 2** (Touvron et al., 2023), **Mistral 7B** (Jiang et al., 2023) 등 주요 오픈소스 LLM에 채택되어, 사실상 현대 LLM의 **표준 어텐션 메커니즘**으로 자리잡았다.

2. **Uptraining 패러다임의 확립**: "기존 체크포인트를 구조적으로 변환 → 소량의 추가 학습"이라는 패러다임은 Mixture-of-Experts 업사이클링(Komatsuzaki et al., 2022)과 함께, **모델 효율화의 비용-효과적 접근법**으로 자리잡았다.

3. **추론 효율성 연구의 가속**: KV 캐시 최적화 연구(PagedAttention, KV cache compression 등)의 근간을 제공.

### 4.2 앞으로 연구 시 고려할 점

1. **Decoder-only 모델에서의 검증**: 원 논문은 encoder-decoder(T5)에서만 실험하였으나, 현재 주류인 decoder-only 모델에서의 체계적 검증이 필요 (이후 LLaMA 2 등에서 이미 일부 검증됨)

2. **최적 그룹 수 $G$의 동적/적응적 결정**: 레이어별로 다른 그룹 수를 사용하거나, 태스크에 따라 적응적으로 결정하는 방법론 연구

3. **Scratch 학습과 Uptraining의 비교**: 동일 계산 예산 하에서 GQA를 처음부터 학습하는 것과 MHA에서 uptraining하는 것의 체계적 비교

4. **KV 캐시 압축과의 결합**: GQA와 양자화(quantization), KV cache eviction 등 다른 효율화 기법의 **상호작용 및 결합 효과** 연구

5. **긴 컨텍스트(Long Context)에서의 영향**: 100K+ 토큰 컨텍스트에서 GQA의 품질-속도 트레이드오프에 대한 심층 분석

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기법 | GQA와의 관계 |
|------|------|----------|-------------|
| **Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need"** | 2019 | Multi-Query Attention (MQA) | GQA의 기반 아이디어. GQA는 MQA의 일반화 |
| **Pope et al., "Efficiently Scaling Transformer Inference"** | 2022 | MQA + 대규모 모델 추론 최적화 | MQA의 대규모 적용, 샤딩 시 비효율 문제를 GQA가 해결 |
| **Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** | 2022 | IO-aware 어텐션 커널 | GQA와 직교적/보완적. FlashAttention + GQA 결합으로 추가 속도 향상 가능 |
| **Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"** | 2023 | FlashAttention 개선 | GQA에 최적화된 커널 구현 지원 |
| **Touvron et al., "LLaMA 2: Open Foundation and Fine-Tuned Chat Models"** | 2023 | **GQA 채택** (34B, 70B 모델) | GQA의 실제 대규모 적용. 34B/70B 모델에서 GQA 사용, 7B/13B는 MHA 유지 |
| **Jiang et al., "Mistral 7B"** | 2023 | GQA + Sliding Window Attention | 7B 규모에서도 GQA 채택, sliding window와 결합 |
| **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"** (vLLM) | 2023 | KV 캐시 페이지 관리 | GQA로 줄어든 KV 캐시를 더 효율적으로 관리하는 보완 기술 |
| **Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"** | 2022 | KV 캐시 양자화 | GQA와 양자화를 결합하여 KV 캐시를 더욱 압축 가능 |
| **Hooper et al., "KVQuant: Towards 10 Million Context Quantization for LLMs"** | 2024 | KV 캐시 초저정밀 양자화 | GQA + 양자화의 시너지로 초대규모 컨텍스트 처리 가능 |
| **Brandon et al., "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention"** | 2024 | 레이어 간 KV 캐시 공유 | GQA의 "헤드 간 공유"를 "레이어 간 공유"로 확장한 연구 |
| **Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints"** | 2022 | Dense → MoE uptraining | GQA의 uptraining 방법론의 영감 원천. 동일한 "체크포인트 구조 변환 + 소량 추가 학습" 패러다임 |

### 핵심 트렌드 분석

**GQA 이후 연구 흐름은 크게 세 방향으로 발전:**

1. **GQA의 채택 및 확산**: LLaMA 2, Mistral, Gemma 등 주요 LLM들이 GQA를 표준으로 채택. 이는 GQA가 이론적 기여를 넘어 **실질적 산업 영향**을 미쳤음을 보여준다.

2. **GQA + 추가 최적화 결합**: FlashAttention-2, PagedAttention, KV 캐시 양자화 등과 GQA를 결합하여 **다중 차원에서의 효율성 향상**을 추구하는 연구가 활발.

3. **KV 캐시 공유의 확장**: GQA가 제안한 "헤드 간 KV 공유"를 **레이어 간 공유**, **시간 축 공유** 등으로 확장하는 후속 연구가 진행 중.

---

## 참고 자료

1. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *arXiv:2305.13245v3*.
2. Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." *arXiv:1911.02150*.
3. Pope, R., Douglas, S., Chowdhery, A., et al. (2022). "Efficiently Scaling Transformer Inference." *arXiv:2211.05102*.
4. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.
5. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *arXiv:2307.08691*.
6. Touvron, H., et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." *arXiv:2307.09288*.
7. Jiang, A. Q., et al. (2023). "Mistral 7B." *arXiv:2310.06825*.
8. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*.
9. Komatsuzaki, A., et al. (2022). "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints." *arXiv:2212.05055*.
10. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*.
11. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR, 21(140):1-67*.
12. Brandon, W., et al. (2024). "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention." *arXiv:2405.12981*.
13. Hooper, C., et al. (2024). "KVQuant: Towards 10 Million Context Quantization for LLMs." *arXiv:2401.18079*.
