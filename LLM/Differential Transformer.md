# Differential Transformer

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 Transformer는 **어텐션 노이즈(attention noise)** 문제, 즉 관련 없는 문맥(irrelevant context)에 과도하게 어텐션 점수를 할당하는 문제를 가지고 있다. DIFF Transformer는 두 개의 softmax 어텐션 맵의 **차분(differential)**을 이용하여 이 노이즈를 상쇄(cancel)하고, 관련 문맥에 대한 어텐션을 증폭(amplify)시키는 새로운 아키텍처를 제안한다.

### 주요 기여
| 기여 항목 | 내용 |
|---|---|
| **차분 어텐션 메커니즘** | 두 softmax 어텐션의 차분으로 노이즈 제거 |
| **스케일링 효율성** | 동등 성능 달성에 모델 크기/토큰 수 약 65%만 필요 |
| **장문맥 모델링** | 64K 컨텍스트에서 더 효과적인 정보 활용 |
| **할루시네이션 완화** | 문맥 기반 할루시네이션 유의미하게 감소 |
| **In-context Learning 향상** | 정확도 및 순서 치환 견고성(robustness) 개선 |
| **활성화 이상값 감소** | 양자화(quantization) 친화적 모델 특성 제공 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 Transformer의 softmax 어텐션은 모든 토큰에 대해 양수(positive) 어텐션 점수를 할당하기 때문에, 관련 없는 문맥 토큰들도 비교적 높은 점수를 받게 된다. 이를 **어텐션 노이즈**라 정의하며, 다음과 같은 실용적 문제를 야기한다:

- **정보 검색 실패**: "Lost in the Middle" 현상 (Liu et al., 2024b) - 핵심 정보가 문맥 중간에 위치할 때 검색 실패
- **할루시네이션**: 올바른 사실이 입력에 있음에도 모델이 부정확한 출력 생성
- **In-context learning 불안정성**: 예시 순서 변경에 따른 성능 변동 큰 문제 (Lu et al., 2022)

---

### 2.2 제안하는 방법 (수식 포함)

#### 차분 어텐션 (Differential Attention)

입력 $X \in \mathbb{R}^{N \times d_{\text{model}}}$에 대해 query, key, value를 다음과 같이 투영한다:

$$[Q_1; Q_2] = XW^Q, \quad [K_1; K_2] = XW^K, \quad V = XW^V$$

여기서 $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times 2d}$이다.

차분 어텐션 연산자는 다음과 같이 정의된다:

$$\text{DiffAttn}(X) = \left(\text{softmax}\!\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right) - \lambda \cdot \text{softmax}\!\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right)\right)V $$

여기서 $\lambda$는 학습 가능한 스칼라로, 학습 동역학 동기화를 위해 다음과 같이 재매개변수화(re-parameterize)된다:

$$\lambda = \exp(\lambda_{q_1} \cdot \lambda_{k_1}) - \exp(\lambda_{q_2} \cdot \lambda_{k_2}) + \lambda_{\text{init}} $$

- $\lambda_{q_1}, \lambda_{k_1}, \lambda_{q_2}, \lambda_{k_2} \in \mathbb{R}^d$: 학습 가능한 벡터
- $\lambda_{\text{init}} \in (0, 1)$: 초기화 상수로, 레이어 인덱스 $l \in [1, L]$에 따라:

$$\lambda_{\text{init}} = 0.8 - 0.6 \times \exp(-0.3 \cdot (l-1))$$

#### 멀티헤드 차분 어텐션 (Multi-Head Differential Attention)

각 헤드 $i \in [1, h]$에 대해:

$$\text{head}_i = \text{DiffAttn}(X;\, W_i^Q, W_i^K, W_i^V, \lambda)$$

$$\overline{\text{head}_i} = (1 - \lambda_{\text{init}}) \cdot \text{LN}(\text{head}_i)$$

$$\text{MultiHead}(X) = \text{Concat}(\overline{\text{head}_1}, \cdots, \overline{\text{head}_h})\, W^O $$

- $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$: 출력 투영 행렬
- $\text{LN}(\cdot)$: 각 헤드에 독립적으로 적용되는 RMSNorm (Headwise GroupNorm)
- $(1 - \lambda_{\text{init}})$: 고정 배율(fixed multiplier)로 Transformer의 그래디언트 흐름과 정렬

> **직관적 이해**: 두 번째 softmax 어텐션 맵이 첫 번째의 "노이즈 추정치"로 작동하며, 두 맵의 차분을 취함으로써 공통 모드 노이즈(common-mode noise)가 상쇄된다. 이는 전기공학의 **차동 증폭기(differential amplifier)**와 **노이즈 캔슬링 헤드폰**의 원리와 유사하다.

---

### 2.3 모델 구조 (Overall Architecture)

전체 아키텍처는 $L$개의 레이어를 쌓으며, 각 레이어는 다음 두 모듈로 구성된다:

$$Y^l = \text{MultiHead}(\text{LN}(X^l)) + X^l $$

$$X^{l+1} = \text{SwiGLU}(\text{LN}(Y^l)) + Y^l $$

- $\text{LN}(\cdot)$: RMSNorm (Zhang & Sennrich, 2019)
- $\text{SwiGLU}(X) = (\text{swish}(XW^G) \odot XW^1)W^2$
- $W^G, W^1 \in \mathbb{R}^{d_{\text{model}} \times \frac{8}{3}d_{\text{model}}}$, $W^2 \in \mathbb{R}^{\frac{8}{3}d_{\text{model}} \times d_{\text{model}}}$

**파라미터 정렬 전략**: DIFF Transformer의 헤드 수 $h = d_{\text{model}} / 2d$ 로 설정하여 (Transformer 헤드 수의 절반), 동일한 파라미터 수와 연산 복잡도(FLOPs)를 유지한다.

**처리량(throughput)**: 동일한 FlashAttention 구현 기준으로 DIFF Transformer는 Transformer 대비 약 6~12% 처리량 감소를 보이며, 이는 허용 가능한 범위로 평가된다.

---

### 2.4 성능 향상

#### 스케일링 성능

| 비교 기준 | DIFF Transformer 효율 |
|---|---|
| 모델 크기 (6.8B vs 11B) | **62.2%** 파라미터로 동등 성능 |
| 모델 크기 (7.8B vs 13.1B) | **59.5%** 파라미터로 동등 성능 |
| 학습 토큰 (160B vs 251B) | **63.7%** 토큰으로 동등 성능 |

#### 하류 태스크 (Downstream Tasks, 3B 모델, 1T 토큰)

| 모델 | ARC-C | ARC-E | BoolQ | HellaSwag | OBQA | PIQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|
| OpenLLaMA-3B-v2 | 33.9 | 67.6 | 65.7 | 70.0 | 26.0 | 76.7 | 62.9 | 57.5 |
| StableLM-3B-v2 | 32.4 | 67.3 | 64.6 | 68.6 | 26.4 | 76.0 | 62.1 | 56.8 |
| **DIFF-3B** | **37.8** | **72.9** | **69.0** | **71.4** | **29.0** | **76.8** | **67.1** | **60.6** |

#### 키 정보 검색 (4K 컨텍스트, N=6, R=2)

$$\text{Transformer: } 0.55 \quad \longrightarrow \quad \text{DIFF Transformer: } 0.85 \quad (+30\%)$$

#### 할루시네이션 평가

| 데이터셋 | Transformer | DIFF |
|---|---|---|
| XSum (요약) | 0.44 | **0.53** |
| CNN/DM (요약) | 0.32 | **0.41** |
| MultiNews (요약) | 0.42 | **0.61** |
| Qasper (QA) | 0.28 | **0.39** |
| HotpotQA (QA) | 0.36 | **0.46** |

#### 수학적 추론 (8개 벤치마크, o1-style)

- 평균 정확도 향상: **+7.5%**
- 개별 벤치마크 전체에서 DIFF Transformer가 우위

#### 활성화 이상값 감소

| 활성화 유형 | Transformer Top-1 | DIFF Top-1 |
|---|---|---|
| Attention Logits | 318.0 | **38.8** |
| Hidden States | 3608.6 | **1688.2** |

4-bit DIFF Transformer가 6-bit Transformer와 동등한 성능 달성.

---

### 2.5 한계점

논문에서 직접적으로 명시된 한계 및 추론 가능한 한계:

1. **처리량 오버헤드**: Transformer 대비 약 6~12%의 처리량 감소가 존재하며, 현재는 범용 FlashAttention2 커널을 사용하여 구현되어 최적화 여지가 있음
2. **멀티모달 미적용**: 실험이 언어 모델링에 한정되어 있으며, 비전 등 다른 모달리티로의 확장은 검증되지 않음
3. **지도 미세조정 후 평가 부재**: 지시 미세조정(instruction tuning) 이후의 종합적인 벤치마크 평가가 제한적임
4. **이론적 최적 $\lambda$ 분석 부재**: $\lambda$의 최적값에 대한 더 깊은 이론적 분석이 필요함
5. **소형 모델(< 1B)에 대한 검증 부재**: 830M이 최소 실험 규모로, 초소형 모델에서의 효과는 미검증

---

## 3. 모델의 일반화 성능 향상 가능성 (심층 분석)

DIFF Transformer의 일반화 성능 향상 가능성은 여러 측면에서 구체적으로 확인된다.

### 3.1 스파스 어텐션 패턴과 일반화의 관계

차분 연산의 핵심 효과는 **희소 어텐션 패턴(sparse attention pattern)**의 자연스러운 출현이다. 수식적으로, 두 softmax 맵의 차분은:

$$A_{\text{diff}} = \text{softmax}\!\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right) - \lambda \cdot \text{softmax}\!\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right)$$

두 맵이 유사한 분포를 공유할 경우 공통 성분이 상쇄되어, 실제로 중요한 토큰 간의 관계만이 두드러지게 된다. 이는 모델이 **본질적 패턴(essential patterns)**에 집중하게 하여 훈련 데이터의 노이즈에 과적합(overfit)하는 경향을 줄인다.

Naderi et al. (2024)의 분석에 따르면, 차분 어텐션은 어텐션 행렬의 **스펙트럼 분포(spectral distribution)**를 더 균형 있게 만들어 **랭크 붕괴(rank collapse)** 문제를 효과적으로 해결한다. 랭크 붕괴는 과적합의 지표 중 하나이므로, 이의 해결은 일반화 성능과 직결된다.

### 3.2 In-Context Learning 견고성: 순서 불변성

일반화 성능의 핵심 지표 중 하나는 **입력 형식의 변화에 대한 견고성**이다. 기존 Transformer는 few-shot 예시의 순서 변경에 매우 민감하게 반응하는 것으로 알려져 있다 (Lu et al., 2022, "Fantastically Ordered Prompts").

TREC 데이터셋에서의 비교:
- 랜덤 배열 시: Transformer 마진 **19.0%** → DIFF 마진 **4.0%** (약 4.75배 견고성 향상)
- 교대 배열 시: Transformer 마진 **56.7%** → DIFF 마진 **13.4%** (약 4.2배 견고성 향상)

이는 DIFF Transformer가 **표면적 패턴(superficial patterns)**보다 **의미적 패턴(semantic patterns)**에 집중함을 시사하며, 이것이 일반화 성능 향상의 근본 원인이다.

### 3.3 Many-shot Learning에서의 일관된 우위

64K 컨텍스트에서 데모 샘플 수를 점진적으로 늘렸을 때, DIFF Transformer는 **모든 데이터셋**에서 **일관되게** 더 높은 정확도와 빠른 수렴을 보인다:

| 데이터셋 | 평균 정확도 향상 |
|---|---|
| TREC (6 classes) | +18.0% |
| TREC-fine (50 classes) | +21.6% |
| Banking-77 (77 classes) | +10.4% |
| Clinic-150 (150 classes) | +5.2% |

클래스 수가 많아질수록 향상폭이 달라지지만, **모든 설정에서 일관된 우위**는 일반화 성능의 강력한 증거다.

### 3.4 도메인 간 일반화: 수학적 추론

언어 모델링 태스크에서 훈련된 기반 위에 수학 데이터로 파인튜닝했을 때, DIFF Transformer는 8개의 다양한 수학 벤치마크 **전체**에서 우위를 보인다. 이는 차분 어텐션 메커니즘이 특정 태스크에 특화된 것이 아니라 **범용적인 정보 처리 능력 향상**을 가져옴을 시사한다.

특히 o1-style 추론에서, DIFF Transformer는 평균 **6,144 토큰**의 추론 과정을 생성한 반면 Transformer는 **6,913 토큰**을 생성했다. 더 짧고 정확한 추론 체인은 모델이 불필요한 정보에 주의를 분산시키지 않음을 보여준다.

### 3.5 장문맥에서의 일반화

64K 토큰 컨텍스트에서의 누적 평균 NLL(Negative Log-Likelihood)이 일관되게 더 낮다는 결과는, 모델이 **긴 거리의 의존성(long-range dependencies)**을 더 잘 포착함을 의미한다. 이는 특히 분포 외(out-of-distribution) 길이의 입력에 대한 일반화 능력과 연관된다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 어텐션 메커니즘 개선 관련 연구

| 연구 | 핵심 아이디어 | DIFF Transformer와의 비교 |
|---|---|---|
| **Sparse Transformer** (Child et al., 2019) | 패턴 기반 희소 어텐션 | DIFF는 **학습**을 통해 자연스럽게 희소성 달성 (하드 제약 없음) |
| **FlashAttention** (Dao et al., 2022) | I/O 효율적 어텐션 구현 | DIFF와 **직접 결합 가능** (상호 보완적) |
| **FlashAttention-2** (Dao, 2023) | 병렬화 개선 | DIFF 구현에 기반으로 활용 |
| **FlashAttention-3** (Shah et al., 2024) | 비동기성 및 저정밀도 활용 | DIFF의 처리량 오버헤드 추가 감소 가능성 |
| **RoPE** (Su et al., 2021) | 회전 위치 임베딩 | DIFF와 **결합하여** 64K 장문맥 확장에 활용 |

### 4.2 어텐션 노이즈 및 할루시네이션 관련 연구

| 연구 | 핵심 내용 | DIFF와의 관련성 |
|---|---|---|
| **Lost in the Middle** (Liu et al., 2024b) | LLM의 장문맥 정보 검색 실패 분석 | DIFF가 이 문제를 근본적으로 해결 |
| **Needle in a Haystack** (Kamradt, 2023) | 장문맥 정보 검색 벤치마크 | DIFF 평가의 핵심 기준으로 활용 |
| **Lookback Lens** (Chuang et al., 2024) | 어텐션 맵으로 할루시네이션 탐지 및 완화 | 어텐션 노이즈-할루시네이션 연관성 지지, DIFF의 할루시네이션 평가 프로토콜로 활용 |
| **OPERA** (Huang et al., 2024) | 멀티모달 LLM 할루시네이션 완화 | 후처리 방식 vs DIFF의 아키텍처 수준 해결 비교 |

### 4.3 효율적 LLM 아키텍처 연구

| 연구 | 핵심 아이디어 | DIFF와의 비교 |
|---|---|---|
| **LLaMA** (Touvron et al., 2023) | 효율적 decoder-only Transformer | DIFF의 기반 아키텍처 (RMSNorm, SwiGLU 채택) |
| **Magneto** (Wang et al., 2023) | Sub-LN 기반 훈련 안정성 | 안정성 접근법 차이: DIFF는 그래디언트 등가성으로 해결 |
| **Mamba/SSM 계열** (2023~) | 선택적 상태 공간 모델 | 어텐션 자체를 대체 vs DIFF는 어텐션 **내부** 개선 |
| **Zoology** (Arora et al., 2023) | 연관 기억(associative recall) 분석 | DIFF의 ablation 평가 지표로 AR-Hit 개념 활용 |
| **StableLM-3B-4E1T** (Tow et al., 2023) | 잘 조율된 Transformer 기준선 | DIFF 비교의 강력한 baseline으로 활용 |

### 4.4 스케일링 법칙 관련

| 연구 | 핵심 내용 | DIFF와의 관련성 |
|---|---|---|
| **Scaling Laws** (Kaplan et al., 2020) | 모델 크기-성능 관계 법칙 | DIFF는 동일 법칙 하에서 더 나은 스케일링 효율 달성 |
| **Gemini 1.5** (Reid et al., 2024) | 백만 토큰 멀티모달 처리 | DIFF의 multi-needle 평가 프로토콜 참고 |

### 4.5 In-Context Learning 연구

| 연구 | 핵심 내용 | DIFF와의 관련성 |
|---|---|---|
| **Fantastically Ordered Prompts** (Lu et al., 2022) | few-shot 순서 민감성 문제 제기 | DIFF가 이 만성적 문제를 아키텍처 수준에서 완화 |
| **In-context learning with long-context** (Bertsch et al., 2024) | 장문맥 ICL 평가 프로토콜 | DIFF의 many-shot ICL 평가에 활용 |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려점

### 5.1 미치는 영향

#### (1) 어텐션 메커니즘 설계 패러다임의 전환
DIFF Transformer는 어텐션 점수의 **절대적 크기** 최적화에서 **신호 대 잡음비(Signal-to-Noise Ratio) 최적화**로의 패러다임 전환을 제시한다. 앞으로의 어텐션 메커니즘 연구는 단순히 계산 효율성뿐 아니라, 어텐션 노이즈 감소를 핵심 설계 원칙으로 고려하게 될 것이다.

#### (2) 어텐션 노이즈의 정량적 지표화
어텐션 노이즈를 명시적으로 정의하고 측정 가능하게 제시함으로써, 향후 연구에서 이를 표준 평가 지표로 활용할 수 있는 기반을 마련하였다.

#### (3) 저비트 어텐션 연산의 가능성
활성화 이상값 감소로 인해 **4-bit 또는 그 이하의 어텐션 커널** 개발 가능성이 열렸다. 이는 LLM 추론 효율화 연구에 새로운 방향을 제시한다.

#### (4) KV 캐시 압축과의 시너지
희소 어텐션 패턴은 KV 캐시 압축(KV cache compression) 연구와 직접적으로 결합 가능하다. 중요하지 않은 토큰의 KV를 더 적극적으로 제거할 수 있어, 장문맥 추론의 메모리 효율을 크게 향상시킬 수 있다.

#### (5) 할루시네이션 연구 방향
아키텍처 수준에서 할루시네이션을 완화한 것은, 기존의 후처리(post-hoc) 방식에 의존하던 연구 흐름에 새로운 방향을 제시한다.

#### (6) 수학적/논리적 추론 향상의 기반
o1-style 추론에서의 유의미한 성능 향상은, 더 강력한 추론 모델 개발에 DIFF 아키텍처가 유용한 기반이 될 수 있음을 시사한다.

---

### 5.2 앞으로 연구 시 고려해야 할 점

#### (1) 멀티모달 확장 검증
현재 실험은 언어 모달리티에 한정되어 있다. 비전-언어 모델, 음성 모델 등에서 차분 어텐션의 효과를 검증하는 연구가 필요하다. 특히 비전 트랜스포머에서 패치 간 어텐션 노이즈 문제가 존재하는지 분석이 필요하다.

#### (2) $\lambda$의 이론적 분석 심화
현재 $\lambda_{\text{init}}$는 경험적으로 결정되었다. 최적 $\lambda$ 값의 이론적 도출, 그리고 $\lambda$가 학습 과정에서 어떻게 수렴하는지에 대한 수학적 분석이 필요하다.

$$\lambda^* = \arg\min_{\lambda} \mathbb{E}[\mathcal{L}(\text{DiffAttn}(X, \lambda))]$$

이 최적화 문제에 대한 이론적 접근이 향후 연구 과제다.

#### (3) 전용 하드웨어 커널 개발
현재 DIFF Transformer는 기존 FlashAttention 커널을 재활용하여 약 6~12%의 처리량 손실이 있다. 차분 어텐션 연산에 특화된 **커스텀 CUDA/Triton 커널** 개발이 실용화의 핵심 과제다.

#### (4) 지시 미세조정 후 포괄적 평가
RLHF, DPO 등의 정렬(alignment) 기법과 결합했을 때의 성능 변화, 그리고 MMLU, MT-Bench 등 표준 지시 모델 벤치마크에서의 평가가 필요하다.

#### (5) 다양한 아키텍처와의 결합 연구
- **MoE(Mixture of Experts)**와 결합 시 효과
- **선택적 상태 공간 모델(SSM)**과 하이브리드 구조에서의 적용
- **Multi-Query Attention (MQA)**, **Grouped-Query Attention (GQA)**와의 결합 가능성

#### (6) 편향성 및 공정성 분석
어텐션 노이즈 감소가 모델의 편향성(bias)에 미치는 영향 분석이 필요하다. 희소 어텐션이 특정 사회적 집단에 대한 표현을 억제하는 부작용이 있을 가능성을 검토해야 한다.

#### (7) 연속 학습(Continual Learning)에서의 효과
어텐션 노이즈 감소가 **치명적 망각(catastrophic forgetting)** 완화에 기여할 수 있는지 검토가 필요하다. 스파스 어텐션 패턴은 더 명확한 표현 분리(representation separation)를 유도할 수 있다.

#### (8) 소형/엣지 디바이스 모델에서의 검증
830M이 최소 실험 규모였으므로, 100M~500M 규모의 소형 모델에서 차분 어텐션의 효과와 오버헤드 비율을 검증하는 연구가 필요하다.

---

## 참고자료

**주요 참고 논문 (본 분석의 근거)**

1. **Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei. "Differential Transformer." Published as a conference paper at ICLR 2025. arXiv:2410.05258v2, 2025.** *(본 답변의 주 출처)*

2. Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.

3. Liu, N. F., et al. "Lost in the middle: How language models use long contexts." TACL, 2024b.

4. Kamradt, G. "Needle in a Haystack - pressure testing LLMs." GitHub, 2023.

5. Dao, T., et al. "FlashAttention: Fast and memory-efficient exact attention with IO-awareness." NeurIPS 2022.

6. Dao, T. "FlashAttention-2: Faster attention with better parallelism and work partitioning." arXiv:2307.08691, 2023.

7. Shah, J., et al. "FlashAttention-3: Fast and accurate attention with asynchrony and low-precision." arXiv:2407.08608, 2024.

8. Touvron, H., et al. "LLaMA: Open and efficient foundation language models." arXiv:2302.13971, 2023.

9. Zhang, B. & Sennrich, R. "Root mean square layer normalization." NeurIPS 2019.

10. Shazeer, N. "GLU Variants Improve Transformer." arXiv:2002.05202, 2020.

11. Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864, 2021.

12. Kaplan, J., et al. "Scaling laws for neural language models." arXiv:2001.08361, 2020.

13. Lu, Y., et al. "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity." ACL 2022.

14. Naderi, A., Nait Saada, T., & Tanner, J. "Mind the gap: a spectral analysis of rank collapse and signal propagation in attention layers." 2024.

15. Chuang, Y.-S., et al. "Lookback Lens: Detecting and mitigating contextual hallucinations in large language models using only attention maps." arXiv:2407.07071, 2024.

16. Wang, H., et al. "Magneto: A Foundation Transformer." ICML 2023.

17. Arora, S., et al. "Zoology: Measuring and improving recall in efficient language models." arXiv:2312.04927, 2023.

18. Bertsch, A., et al. "In-context learning with long-context models: An in-depth exploration." arXiv:2405.00200, 2024.

19. Guo, D., et al. "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning." arXiv:2501.12948, 2025.

20. Tow, J., et al. "StableLM 3B 4E1T." Technical Report, 2023.

21. Laplante, P. A., et al. "Comprehensive dictionary of electrical engineering." CRC Press, 2018.

22. Wu, Y. & He, K. "Group Normalization." ECCV 2018.

23. Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019.
