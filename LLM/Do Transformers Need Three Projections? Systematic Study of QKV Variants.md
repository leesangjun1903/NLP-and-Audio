# Do Transformers Need Three Projections? Systematic Study of QKV Variants

---

## 📌 참고 자료

- **주 논문**: Kayyam, A., Madan Gopal, A., & Lewis, M. A. (2026). *Do Transformers Need Three Projections? Systematic Study of QKV Variants*. Proceedings of the 43rd ICML. arXiv:2606.04032v2
- **관련 논문**: Vaswani et al. (2017). *Attention is All You Need*. NeurIPS.
- **관련 논문**: Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models*. EMNLP.
- **관련 논문**: Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need*. arXiv:1911.02150
- **관련 논문**: Liu et al. (2024). *DeepSeek-V2*. arXiv:2405.04434
- **관련 논문**: He & Hofmann (2024). *Simplifying Transformer Blocks*. ICLR.
- **관련 논문**: Kowsher et al. (2025). *Does Self-Attention Need Separate Weights in Transformers?* NAACL.
- **관련 논문**: Hwa, Holmes & Drechsler (2025). *Integration of Key-Value Attention into Pure and Hybrid Transformers for Semantic Segmentation*. BVM Workshop.
- **관련 논문**: Katharopoulos et al. (2020). *Transformers are RNNs*. ICML.
- **관련 논문**: Gu & Dao (2023). *Mamba*. arXiv:2312.00752

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

트랜스포머의 Query(Q), Key(K), Value(V) 세 개의 독립 projection이 **모두 필수적이지 않다**. 특히 **K와 V를 공유(Q-K=V)** 하는 방식은 50% KV 캐시 감소와 함께 성능 손실을 최소화하며, 이는 엣지 디바이스 배포에 실용적인 대안이 된다.

### 🏆 주요 기여 5가지

| 기여 항목 | 내용 |
|-----------|------|
| 체계적 평가 | 12개 다양한 태스크(합성 추론, 컴퓨터 비전, LLM)에서 projection sharing 전략 벤치마크 |
| 캐시 최적화 | Q-K=V가 KV 캐시 50% 감소, perplexity 단 3.1% 증가 |
| 스케일 검증 | 1.2B 파라미터까지 상대적 성능 순위 안정적으로 유지됨 확인 |
| 구조적 시너지 | Projection sharing + Head sharing (GQA/MQA) 조합으로 최대 96.9% 캐시 감소 |
| 이론적 통찰 | Q-K=V 작동 원리(저차원 표현 공간 공유)와 Q=K-V 실패 원인(방향성 파괴) 분석 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

트랜스포머는 각 토큰에 대해 Q, K, V 세 개의 독립적인 선형 투영을 유지하는데, 이는:

1. **KV 캐시 메모리 병목**: 긴 컨텍스트나 고처리량 서빙 시나리오에서 메모리 지배
2. **파라미터 중복성**: K와 V 행렬 간 표현 공간의 실질적 중복 존재 가능성
3. **엣지 배포 제약**: 온디바이스 추론에서 메모리 한계로 인한 실용성 부족

> **핵심 질문**: "트랜스포머는 정말 세 개의 독립적인 projection이 필요한가?"

---

### 2.2 제안하는 방법 (수식 포함)

#### 표준 Attention (QKV 기준선)

입력 $X \in \mathbb{R}^{n \times d}$에 대해 각 헤드 $h$의 어텐션:

$$A_h = \text{Softmax}\left(\alpha Q_h K_h^T\right)V_h \tag{1}$$

여기서 $Q_h = XW_q$, $K_h = XW_k$, $V_h = XW_v$, 스케일 팩터 $\alpha = \frac{1}{\sqrt{d_k}}$

---

#### Variant 1: Q=K-V (Query-Key 공유, Value 분리)

$$A = \text{Softmax}\left(\alpha K K^T\right)V \tag{2}$$

- **특성**: $KK^T$는 대칭 attention 행렬 생성
- **문제**: 순서 방향성 파괴 → 인과적 언어 모델링에 불리
- **해결**: $(Q=K\text{-}V)^+$ — 2D 위치 인코딩 $P \in \mathbb{R}^{n \times n \times m}$ 주입으로 비대칭성 복구

$$A' = A + P, \quad A' \in \mathbb{R}^{n \times n \times m} \xrightarrow{1\times1 \text{ conv}} \mathbb{R}^{n \times n}$$

---

#### Variant 2: Q-K=V (Key-Value 공유, Query 분리) ⭐

$$A = \text{Softmax}\left(\alpha Q K^T\right)K \tag{3}$$

- **특성**: $Q$와 $K$가 독립적 → **비대칭 attention 맵 유지**
- **장점**: 자동회귀 생성 시 $K$ 텐서만 캐싱 → **50% KV 캐시 감소**
- **해석**: weight tying의 한 형태 (Press & Wolf, 2017)

---

#### Variant 3: Q=K=V (단일 projection)

$$A = \text{Softmax}\left(\alpha K K^T\right)K \tag{4}$$

- **특성**: 가장 공격적인 단순화, 파라미터 $\frac{1}{3}$ 수준
- **문제**: 대칭 attention + 표현 병목 → 언어 모델링에서 **25.4% perplexity 급증**

---

#### 2D 위치 인코딩의 선형 Attention 적용 시 (Appendix A.1)

$q_t = k_t = v_t = z_t$ (QKV 붕괴)일 때 선형 어텐션:

$$y_t = \frac{\phi(q_t)^\top \sum_{i \leq t} \phi(k_i)v_i^\top}{\phi(q_t)^\top \sum_{i \leq t} \phi(k_i)} \tag{5}$$

상태 업데이트:

$$S_t = S_{t-1} + \phi(z_t)z_t^\top \tag{7}$$

이는 고전적 SSM의 상태 방정식과 대응:

$$h_t = Ah_{t-1} + Bx_t, \quad y_t = Ch_t \tag{8}$$

→ **Q=K=V 하의 선형 어텐션은 적응적 관측을 가진 상태공간 모델의 특수 케이스**

---

### 2.3 모델 구조

#### 기본 아키텍처 (LLM 실험)

| 항목 | 300M 모델 | 1.2B 모델 |
|------|-----------|-----------|
| 레이어 수 | 20 | 22 |
| 임베딩 차원 $d$ | 1,024 | 2,048 |
| 어텐션 헤드 수 | 16 | 32 |
| MLP 차원 | 4,096 | 8,192 |
| 어휘 크기 | 50,304 | 50,304 |

- **활성화 함수**: GELU
- **정규화**: Pre-Norm LayerNorm ($\epsilon = 10^{-5}$)
- **위치 인코딩**: 학습된 절대 위치 임베딩 (최대 2048 토큰)
- **옵티마이저**: AdamW ($\beta_1=0.9$, $\beta_2=0.95$, weight decay $0.1$)

#### 계산 복잡도 비교 (projection 연산 기준)

| 변형 | 계산량 | 파라미터 수 |
|------|--------|------------|
| QKV (기준) | $3nd^2$ | $3d^2$ |
| Q=K-V / Q-K=V | $2nd^2$ | $2d^2$ |
| $(Q=K\text{-}V)^+$ | $2nd^2 + n^2m$ | $2d^2 + m$ |
| Q=K=V | $nd^2$ | $d^2$ |

> 주의: $O(n^2d)$의 어텐션 스코어 계산 비용은 모든 변형에서 공유

---

### 2.4 성능 향상

#### 합성 태스크 결과 (정확도 평균)

| 모델 | REVERSE | SORT | SUB | SWAP | COPY | **Avg** |
|------|---------|------|-----|------|------|---------|
| QKV | 0.698 | 0.971 | 1.0 | 0.588 | 1.0 | 0.851 |
| Q=K-V | 0.705 | 0.967 | 1.0 | 0.597 | 1.0 | 0.854 |
| $(Q=K\text{-}V)^+$ | 0.718 | 0.963 | 1.0 | 0.671 | 1.0 | **0.870** |
| Q-K=V | 0.701 | 0.958 | 1.0 | 0.590 | 1.0 | 0.850 |
| Q=K=V | 0.514 | 0.939 | 1.0 | 0.446 | 1.0 | 0.780 |

#### 언어 모델링 결과 (300M 파라미터)

| 모델 | Val PPL | PPL 증가율 | KV 캐시 감소 |
|------|---------|-----------|------------|
| QKV (기준) | 5.11 | 0% | 0% |
| **Q-K=V** | **5.27** | **+3.1%** | **50%** |
| Q=K-V | 5.36 | +4.9% | 0% |
| Q=K=V | 6.41 | +25.4% | 50% |
| GQA-4 | 5.15 | +0.7% | 75% |
| MQA | 5.19 | +1.5% | 93.8% |
| **Q-GQA-4** | **5.32** | **+3.9%** | **87.5%** |
| **Q-MQA** | **5.36** | **+4.8%** | **96.9%** |

#### 1.2B 스케일 다운스트림 태스크 (5-shot, Table 11 기반)

| 모델 | ARC-C | ARC-E | HellaSwag | PIQA | WinoG | **Avg** | Cache↓ |
|------|-------|-------|-----------|------|-------|---------|--------|
| QKV | 19.03 | 30.01 | 26.15 | 56.64 | 50.20 | 36.40 | — |
| Q-K=V | 18.94 | 28.62 | 26.14 | 56.37 | 49.88 | 35.99 | 50% |
| Q-GQA-8 | 19.97 | 29.97 | 26.13 | 56.47 | 51.07 | **36.72** | 87.5% |

> Q-K=V는 perplexity 2.48% 증가에도 불구하고 다운스트림 태스크 평균 성능 손실은 **0.41%**에 불과 — perplexity 열화와 실제 태스크 성능 간의 **디커플링** 현상 확인

---

### 2.5 한계점

1. **최대 검증 규모**: 1.2B 파라미터; 7B+ 규모에서의 경향 미확인
2. **이론적 형식화 부재**: Q-K=V 성능 보존 이유가 실증적 설명에 그침
3. **시퀀스 길이 제한**: 2048 토큰까지만 평가; 길이 외삽(extrapolation) 미검토
4. **Q=V 변형 미포함**: Q는 생성 시 캐싱 안 됨 → 분석 범위에서 자연스럽게 제외
5. **단일 데이터셋**: LLM 실험이 SlimPajama 하나에 국한

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 일반화 관련 실증 근거

#### (1) 퍼플렉시티-다운스트림 태스크 디커플링

Q-K=V 모델의 경우:

$$\Delta\text{PPL} = +2.48\% \quad \text{(1.2B 스케일)} \quad \longleftrightarrow \quad \Delta\text{Task Acc} = -0.41\%$$

이 **비례하지 않는 격차**는 perplexity가 일반화 능력의 완전한 대리 지표가 아님을 시사하며, projection sharing이 실질적 일반화 능력을 크게 손상시키지 않음을 보여준다.

#### (2) Weight Tying 효과를 통한 정규화

$K = V$ 제약은 일종의 **파라미터 정규화(regularization)** 로 작용한다:

$$W_k = W_v \quad \Rightarrow \quad \text{파라미터 공간 축소} \quad \Rightarrow \quad \text{과적합 위험 감소}$$

Press & Wolf (2017)의 출력 임베딩 weight tying이 언어 모델 성능을 향상시켰듯, K=V tying도 유사한 정규화 효과를 낼 수 있다.

#### (3) K와 V의 표현 공간 유사성 (실증 분석)

학습된 QKV 모델 분석 결과:

| 행렬 쌍 | 코사인 유사도 | 유효 랭크 (1024차원 중) |
|---------|-------------|----------------------|
| K와 V | **0.73** | 687 vs 702 |
| Q와 K | 0.42 | — |
| Q와 V | 0.31 | — |

K와 V는 **높은 코사인 유사도(0.73)**와 **유사한 유효 랭크**를 보이며, K=V 공유가 표현력 손실을 최소화함을 보여준다. 반면 Q는 K, V 모두와 낮은 유사도 → **방향성 어텐션에 필수적인 비대칭성 유지**.

#### (4) 저랭크 어텐션 체제 (Low-Rank Regime)

어텐션 행렬이 저랭크 구조로 수렴하는 경향이 있어, K=V 제약이 부과하는 표현 병목이 실질적 정보 손실을 거의 유발하지 않는다.

#### (5) 스케일에 따른 일반화 개선 경향

| 스케일 | Q-K=V PPL 열화 |
|--------|--------------|
| 300M | +3.1% |
| 1.2B | +2.48% |

**더 큰 모델에서 열화가 감소**하는 경향 → 7B+ 규모에서 projection sharing이 더 유리해질 가능성 시사.

#### (6) 시퀀스 길이 별 열화 추이 (Table 16)

| 시퀀스 길이 | Q-K=V PPL 열화 |
|-----------|--------------|
| 512 토큰 | +5.4% |
| 1024 토큰 | +3.7% |
| 2048 토큰 | +2.2% |

**긴 컨텍스트에서 상대적 품질 손실 감소** → 장문 컨텍스트 일반화에 유리.

#### (7) 도메인 일반화 (다중 태스크)

비전(MNIST, CIFAR, TinyImageNet, 이상 탐지), 합성 추론, 의료 영상 분할(Hwa et al., 2025)에 이르기까지 **12개 다양한 태스크**에서 성능 유지 확인 → 도메인 횡단 일반화 성능 입증.

### 3.2 일반화 성능 향상 메커니즘 요약

```
[일반화 성능 향상 경로]

K=V 제약
    ├─ 파라미터 수 감소 → 과적합 위험 완화 (weight tying 효과)
    ├─ 표현 공간 정규화 → 일반화 가능한 특성 학습 유도
    ├─ 저랭크 어텐션 활용 → 노이즈 필터링 효과
    └─ 스케일 증가 시 열화 감소 → 대형 모델에서 더욱 유리
```

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) Attention 설계 패러다임의 재정립

이 논문은 "Q, K, V 세 projection이 모두 필수적"이라는 암묵적 가정에 **최초의 체계적 반례**를 제시한다. 향후 연구들은 projection 설계를 자유 변수로 다룰 가능성이 높아진다.

#### (2) 효율성 연구의 새로운 축 제시

기존 효율화 연구가 주로 head sharing(GQA/MQA)이나 linear attention에 집중했다면, 이 논문은 **projection sharing**이라는 **직교하는 새 축**을 제시한다:

$$\text{Total Cache Reduction} = 1 - \left(1 - \frac{g}{H}\right)\left(1 - \frac{1}{2}\right) \quad \text{(Q-GQA-g의 경우)}$$

#### (3) 선형 어텐션과 SSM 통합 이론 기여

Appendix A.1에서 Q=K=V 붕괴 하에 선형 어텐션이 SSM의 특수 케이스임을 증명함으로써, **Transformer ↔ RNN ↔ SSM** 통합 프레임워크 연구에 이론적 토대를 제공한다.

#### (4) 엣지/온디바이스 AI 실용화 촉진

| 시나리오 | 권장 구성 | 캐시 감소 | PPL 열화 |
|---------|---------|---------|---------|
| 클라우드 (품질 우선) | GQA-4 | 75% | +0.7% |
| 엣지 (균형) | Q-K=V | 50% | +3.1% |
| 엣지 (공격적) | Q-GQA-4 | 87.5% | +3.9% |
| IoT/모바일 | Q-MQA | 96.9% | +4.8% |

#### (5) Weight Tying 연구의 확장

Press & Wolf (2017)의 임베딩 weight tying → 이 논문의 attention projection tying으로 **weight sharing의 적용 범위를 확장**했으며, 이는 Transformer의 다른 구성 요소(FFN, LayerNorm 등)로의 확장 연구를 자극할 것이다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 더 큰 스케일 검증 필요

```
현재: 300M → 1.2B (열화 감소 경향 확인)
필요: 7B, 13B, 70B+ 규모에서 경향 지속 여부 검증
```

특히 1.2B에서 Q-K=V의 열화가 2.48%로 줄었다면, 7B+에서 1% 이하로 수렴할 가능성을 체계적으로 검증해야 한다.

#### (2) 더 긴 컨텍스트 평가

현재 최대 2048 토큰까지만 평가됨. 32K, 128K 컨텍스트에서:
- KV 캐시 절감 효과는 선형적으로 증가 (더 유리)
- 하지만 주의 패턴의 장거리 의존성에 K=V 제약이 미치는 영향 불명확

#### (3) Fine-tuning 시나리오 검토

현재 연구는 **처음부터 학습(from scratch)**에 집중. 실제 산업 환경에서 중요한:
- 기존 QKV 모델을 Q-K=V로 변환 후 fine-tuning하는 경우의 성능
- RLHF, instruction tuning 후 Q-K=V 유지 여부

#### (4) Q=V 변형 탐색

논문에서 명시적으로 제외된 $Q = V$ 변형:

$$A = \text{Softmax}\left(\alpha Q K^T\right)Q$$

Q는 캐싱되지 않으므로 메모리 이점은 없지만, 표현력 측면에서 흥미로운 비교가 될 수 있다.

#### (5) 이론적 분석 심화

현재 K와 V의 표현 공간 유사성에 대한 설명이 **실증적(empirical)**에 그침. 향후 연구에서:

$$\cos(W_k, W_v) = 0.73 \quad \Rightarrow \quad \text{왜 학습이 이 방향으로 수렴하는가?}$$

의 이론적 해명이 필요하다. 이는 최적화 이론과 정보 이론적 접근을 통해 탐구 가능하다.

#### (6) 다양한 아키텍처로의 일반화

- **Encoder-only**: BERT 계열에서의 Q-K=V 성능 (양방향 어텐션 → 인과성 문제 없음)
- **Encoder-Decoder**: cross-attention 층에 적용 가능성 (self-attention과 cross-attention 분리 전략)
- **Sparse Transformer, Longformer**: 희소 어텐션과의 조합 효과

#### (7) 하드웨어 최적화와의 결합

```
Q-K=V + FlashAttention + INT8 Quantization
= 50% (KV cache) × 50% (INT8) × FlashAttention 속도 향상
= 복합적 메모리/연산 최적화
```

이러한 조합의 **실제 하드웨어 구현 최적화 연구**가 필요하다.

#### (8) 멀티모달 적용

비전-언어 모델(VLM), 음성-텍스트 모델 등에서 cross-modal attention에 Q-K=V 적용 시 성능 변화 탐구.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법 | 핵심 차이 | 본 논문과의 관계 |
|------|------|----------|---------------|
| **GQA** (Ainslie et al., 2023, EMNLP) | 여러 Q 헤드가 공유된 K,V 헤드 그룹 참조 | **Head** 수준 공유 | 직교적 — 조합 가능 (Q-GQA) |
| **MQA** (Shazeer, 2019) | 단일 K,V 헤드로 모든 Q 서빙 | **Head** 수준 극단적 공유 | 직교적 — 조합 가능 (Q-MQA) |
| **DeepSeek-V2 MLA** (Liu et al., 2024) | K,V를 압축된 잠재 벡터로 캐싱 후 추론 시 확장 | 추가 projection 파라미터로 **학습된 압축** | 소프트 제약 vs. Q-K=V의 하드 등호 제약 |
| **Mamba** (Gu & Dao, 2023) | 선택적 상태공간 모델, 어텐션 없이 선형 복잡도 | 어텐션 메커니즘 자체를 대체 | 본 논문은 어텐션 유지하며 projection 공유 |
| **Simplifying Transformer Blocks** (He & Hofmann, 2024, ICLR) | Skip connection, value projection 등 여러 구성요소 단순화 | V projection이 덜 중요함을 독립적으로 확인 | 상호 보완적 실증 — Q-K=V 정당성 지지 |
| **Kowsher et al.** (2025, NAACL) | 유사한 projection 공유 접근 | 독립적 제안 | 본 논문이 더 광범위한 평가 및 LLM 스케일 검증 포함 |
| **Linear Attention** (Katharopoulos et al., 2020, ICML) | Softmax를 커널 함수로 대체 → 선형 복잡도 | 어텐션 계산 방식 변경 | 본 논문과 직교, Q=K=V시 SSM과 연결됨을 이론적으로 증명 |
| **FlashAttention** (Dao et al., 2022, NeurIPS) | IO-aware 정확한 어텐션 구현 | 하드웨어 수준 최적화 | 조합 가능 — Q-K=V + FlashAttention 시너지 |

### 본 논문의 차별점 요약

$$\underbrace{\text{GQA/MQA}}_{\text{Head 수 축소}} \perp \underbrace{\text{Q-K=V}}_{\text{Projection 행렬 공유}} \Rightarrow \underbrace{\text{Q-GQA/Q-MQA}}_{\text{복합 최적화: 최대 96.9\% 캐시 감소}}$$

기존 연구들이 **head 수준** 또는 **어텐션 계산 방식** 변경에 집중한 반면, 본 논문은 **projection 행렬 수준**에서의 공유라는 독립적이고 직교적인 차원을 체계적으로 탐구한 첫 번째 대규모 실증 연구다.

---

> **⚠️ 주의**: 본 논문은 2026년 ICML에 발표 예정이며 (arXiv 기준 2025년 6월), 일부 관련 연구(Kowsher et al. 2025 등)와의 선후 관계 및 독립적 제안 여부는 논문 내 각주를 통해 명시적으로 언급되어 있습니다. 2020년 이후 비교 분석은 본 논문이 인용한 문헌을 중심으로 작성되었으며, 본 논문 제출 이후 등장한 후속 연구와의 비교는 포함되지 않았습니다.
