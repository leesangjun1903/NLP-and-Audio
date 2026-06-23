# Multi-Turn Reasoning When Context Arrives in Pieces: Scalable Sharding and Memory-Augmented RL

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **Lost in Conversation(LiC)** 현상, 즉 사용자가 대화의 여러 턴에 걸쳐 과제에 필요한 정보를 점진적으로 공개할 때 LLM의 정확도가 최대 **65%까지 하락**하는 현상을 다룹니다. 저자들은 이 문제를 해결하기 위해 모델이 전체 대화 히스토리를 참조하는 대신, **압축된 롤링 메모리(rolling memory)**를 유지하도록 훈련하는 방법이 효과적임을 주장합니다.

### 세 가지 주요 기여

| 기여 | 내용 |
|------|------|
| **(1) 저비용 샤딩 파이프라인** | 단일 턴 QA 데이터셋을 멀티턴 에피소드로 자동 변환 (1~3개 few-shot 예시만 필요) |
| **(2) 메모리 증강 RL 레시피** | GSM8K 기준 LiC 손실 최대 60pp 회복, Full-History 기반 훈련 대비 우수한 성능 |
| **(3) 도메인 일반화 증거** | GSM8K만으로 훈련 후 MATH500, LongBench에 제로샷 전이 성공 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**Lost in Conversation(LiC)** 현상은 다음 두 가지 근본적 어려움을 포함합니다:

1. **데이터 희소성**: 기존 LiC 벤치마크(Laban et al., 2026)는 태스크당 1~4시간의 수동 검수와 LLM 시뮬레이션 30회를 요구하며, 최종적으로 90~120개의 검증 예시만 생성됩니다. 이는 RLVR 훈련에 턱없이 부족합니다.

2. **전체 히스토리 의존 문제**: 모델이 기본적으로 전체 대화 히스토리를 참조하는 방식은 컨텍스트 길이가 증가할수록 성능이 저하되는 **"Lost in the Middle"** 현상(Liu et al., 2023)을 유발합니다.

---

### 2.2 제안하는 방법

#### (A) 샤딩 파이프라인 (Sharded Dataset Construction)

단일 턴 QA 문제를 LLM 기반 3단계 파이프라인으로 분해합니다:

1. **분할(Split)**: 문제를 최소 논리 단위로 분할
2. **검증(Verify)**: 완결성 및 비중복성 검증
3. **추출(Extract)**: 핵심 질문을 분리하여 매 턴마다 함께 제공

세 가지 데이터셋 변형을 구성합니다:
- **Sharded**: 논리적 분해
- **Full-Question**: 무작위 위치에 전체 질문 + 나머지는 Wikitext 노이즈
- **Mixed**: Sharded + Wikitext 노이즈 혼합

#### (B) 메모리 메커니즘 (Memory Mechanism)

각 샤드 턴 $t \in \{1, \ldots, K\}$에서, 정책은 다음을 입력으로 받아 메모리를 갱신합니다:

$$m_{t+1} \sim \pi_\theta(\cdot \mid q, m_t, s_t)$$

여기서:
- $q$: 메인 질문
- $m_t$: 현재 메모리 상태 (최대 $L_m = 256$ 토큰으로 제한)
- $s_t$: 현재 샤드

최종 턴에서 모델은 질문 $q$와 최종 메모리 $m_K$만을 조건으로 답변을 생성합니다:

$$a \sim \pi_\theta(\cdot \mid q, m_K)$$

> **핵심**: 메모리는 직접적인 지도 학습 신호 없이, **다운스트림 보상 신호만으로** 그 구조가 형성됩니다.

#### (C) 훈련 목적 함수 (Training Objective)

각 샘플에 대해 $G$개의 궤적을 샘플링하고, 규칙 기반 검증기로 스칼라 보상 $R^{(g)} = \mathcal{V}(a^{(g)}, y^\star)$를 산출합니다.

**그룹 상대적 이점(Group-Relative Advantage)**:

$$\hat{A}^{(g)} = R^{(g)} - \frac{1}{G} \sum_{h=1}^{G} R^{(h)}$$

이 이점은 궤적 $g$의 **모든 턴에 균등하게 전파**됩니다(중간 크레딧 할당 회피).

**최종 손실 함수** (DAPO 기반 이중 클리핑 + KL 패널티):

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{clip}}(\theta) + \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

여기서 $\beta = 0.001$이며, 토큰 평균 집계(token-mean aggregation)를 통해 긴 메모리 업데이트가 기울기를 지배하지 않도록 제어합니다.

---

### 2.3 모델 구조

```
[단일 턴 QA] → LLM 샤딩 파이프라인 → [멀티턴 샤드 에피소드]
                                               ↓
                              [메모리 증강 정책 πθ 훈련]
                              ┌─────────────────────────────┐
                              │ Turn 1: (q, m₀, s₁) → m₁  │
                              │ Turn 2: (q, m₁, s₂) → m₂  │
                              │    ...                      │
                              │ Turn K: (q, m_{K-1}, sK)→mK│
                              │ Final:  (q, mK) → answer   │
                              └─────────────────────────────┘
                                               ↓
                              RLVR (multi-conv DAPO) 업데이트
```

**훈련 하이퍼파라미터**:

| 파라미터 | 값 |
|---------|-----|
| 훈련 에폭 | 10 |
| 배치 크기 | 64 |
| 최대 메모리 버퍼 $L_m$ | 256 tokens |
| 최대 샤드/에피소드 | 35 |
| 최대 응답 길이 | 512 (1.5B) / 1,024 (4B) tokens |
| KL 패널티 $\beta$ | 0.001 |
| 학습률 | $1 \times 10^{-6}$ |
| GPU | 2×A100 |

---

### 2.4 성능 향상

#### 메모리 RL vs. 기준 모델

| 모델 | 방법 | GSM8K | MATH500 | Noisy (n=2) | Noisy (n=4) |
|------|------|-------|---------|-------------|-------------|
| Qwen2.5-1.5B | Base | 0.199 | 0.334 | 0.290 | 0.284 |
| Qwen2.5-1.5B | Memory RL | **0.799** | **0.550** | 0.290 | 0.290 |
| Qwen3-4B (r=1024) | Base | 0.370 | 0.260 | 0.232 | 0.196 |
| Qwen3-4B (r=1024) | Memory RL (Sharded) | **0.877** | **0.638** | 0.538 | 0.438 |
| Qwen3-4B (r=1024) | Memory RL (Mixed) | 0.849 | **0.692** | **0.630** | **0.570** |

#### LongBench F1 (4B, OOD 평가)

| 모델 | 2Wiki | Hotpot | Multi | Qasper | Trivia |
|------|-------|--------|-------|--------|--------|
| Base | 0.470 | 0.407 | 0.260 | 0.256 | 0.691 |
| Sharded | 0.682 | 0.588 | 0.392 | 0.307 | 0.775 |
| Mixed | **0.713** | **0.627** | 0.406 | **0.390** | **0.852** |

**주요 발견**:
- LiC 손실 최대 **60pp 회복** (GSM8K, 1.5B 기준: 0.20 → 0.80)
- Memory RL은 Full-History 조건에서도 Full-History RL보다 **15.2~35.6pp 우수**
- 노이즈 샤드 주입 시 Full-History RL은 급격히 저하되지만, Memory RL은 상대적으로 견고

---

### 2.5 한계점

1. **단일 모델 패밀리**: Qwen 계열만 실험하여 타 모델 패밀리 검증 미완
2. **메모리 버퍼 크기 고정**: 256 토큰 제한이 일부 복잡한 태스크에서 정보 손실 유발 가능
3. **응답 길이 민감성**: Qwen3-4B의 경우 $r=512$에서 MATH500 정확도가 0.050으로 붕괴; $r=1024$에서 회복 → 응답 길이 설정이 사고 모델에 매우 중요
4. **도메인 매칭 없는 평가**: LongBench는 태스크별 훈련 없이 평가되어 최대 잠재 성능 미반영
5. **계산 비용**: 수렴까지 약 30~35시간/run (2×A100)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 제로샷 전이 성능의 의미

논문의 가장 중요한 발견 중 하나는 **GSM8K만으로 훈련된 모델이 MATH500과 LongBench에 제로샷으로 전이**된다는 점입니다. 이는 메모리 압축 훈련이 단순한 데이터셋 특화 패턴 매칭이 아니라, **도메인 불가지론적 점진적 추론 능력(domain-agnostic incremental reasoning)**을 유도함을 시사합니다.

### 3.2 일반화 메커니즘 분석

#### (A) 노이즈 필터링 능력의 전이

$$\text{Mixed 변형} > \text{Full-Question 변형} > \text{Sharded 변형}$$

이 순서는 LongBench의 모든 서브셋에서 일관되게 관찰됩니다. 훈련 중 노이즈 컨텍스트에 노출될수록 **주의분산 필터링(distractor-filtering)** 능력이 향상되고, 이 능력이 개방형 QA로 전이됩니다.

#### (B) 메모리 압축이 유도하는 표현 robustness

Memory RL 모델은 추론 시 Full-History 조건(메모리 없이 전체 히스토리 제공)으로 평가해도 Full-History RL 모델보다 우수합니다. 이는:

$$\text{Memory RL (Full-History 평가)} > \text{Full-History RL (Full-History 평가)}$$

즉, 메모리 압축 학습이 **추론 시 압축을 활용하는 것 이상의** 더 강건한 내부 표현을 형성함을 의미합니다.

#### (C) 일반화를 위한 최적 훈련 변형

| 목표 | 권장 변형 |
|------|---------|
| 인도메인 정확도 최대화 | Sharded |
| 노이즈 견고성 + OOD 전이 | Mixed |
| 전반적 균형 | Mixed |

#### (D) 응답 길이와 사고 능력의 관계

사고 모델(Qwen3-4B)에서 $r=512$ 토큰 제한은 MATH500 정확도를 0.050으로 붕괴시키지만, $r=1024$에서 0.638로 회복됩니다. 이는 **충분한 Chain-of-Thought 공간이 일반화에 필수적**임을 보여줍니다:

$$\text{일반화 성능} \propto \text{응답 길이 예산 (사고 모델의 경우)}$$

### 3.3 일반화 가능성의 잠재적 확장

1. **다양한 도메인 few-shot 예시 교체**: 샤딩 프롬프트의 few-shot 예시를 코드, 과학, 법률 등 도메인별로 교체하면 해당 도메인으로의 전이 가능성이 높음
2. **더 큰 메모리 버퍼**: $L_m > 256$으로 완화하면 복잡한 멀티홉 추론에서 추가 이득 기대
3. **다국어 전이**: 샤딩 파이프라인이 언어 독립적 구조를 활용하므로 다국어 설정에서의 전이 가능성 존재

---

## 4. 미래 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### (A) 패러다임 전환: 컨텍스트 압축 학습의 가치

본 논문은 "더 많은 컨텍스트 = 더 나은 성능"이라는 통념에 도전합니다. **학습을 통한 선택적 압축**이 전체 컨텍스트 노출보다 우수한 추론 품질을 유도할 수 있음을 실증합니다. 이는 LLM 설계 철학에 근본적인 질문을 제기합니다.

#### (B) 데이터 생성의 민주화

1~3개의 few-shot 예시만으로 임의의 단일 턴 QA를 멀티턴 훈련 데이터로 변환하는 파이프라인은, **소규모 연구 그룹도 멀티턴 RL 훈련**을 수행할 수 있는 기반을 제공합니다.

#### (C) RLVR의 적용 범위 확장

기존 RLVR 연구(DeepSeek-R1 등)가 주로 단일 턴 추론에 집중했다면, 본 논문은 이를 **진정한 멀티턴 대화 설정**으로 확장하는 구체적 방법론을 제시합니다.

### 4.2 향후 연구 시 고려할 점

#### (A) 즉각적 후속 연구 방향

| 연구 방향 | 구체적 내용 |
|----------|-----------|
| **크로스 패밀리 검증** | Llama, Mistral, GPT 등 다른 모델 패밀리 적용 |
| **동적 메모리 버퍼** | 고정 256 토큰 대신 정보량에 따른 적응형 버퍼 크기 |
| **계층적 메모리** | 단기/장기 메모리 분리 구조 도입 |
| **실제 대화 데이터** | 합성 샤드가 아닌 실제 멀티턴 대화 적용 |

#### (B) 기술적 고려사항

1. **중간 단계 크레딧 할당**: 현재는 이점을 모든 턴에 균등 전파하지만, 더 정교한 크레딧 할당이 성능을 향상시킬 수 있습니다:

$$\hat{A}^{(g)}_t = f(R^{(g)}, t, K) \quad \text{(턴별 차등 가중치)}$$

2. **메모리 검증 가능성**: 현재 메모리는 자유형 자연어로, 그 품질을 평가하기 어렵습니다. 구조화된 메모리 형식(JSON, 키-값 등)과의 비교 연구가 필요합니다.

3. **확장성 문제**: 에피소드당 최대 35개 샤드 제한이 실제 장문 대화(수백 턴)에는 적용 불가능합니다.

#### (C) 평가 방법론 개선

- 현재 LongBench 평가는 도메인 매칭 훈련 없이 이루어져 최대 잠재 성능을 반영하지 못합니다.
- LiC 벤치마크(Laban et al., 2026)의 실제 인간 대화 설정에서의 평가가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표 연도 | 핵심 방법 | LiC 해결 | 확장성 | 일반화 |
|------|---------|---------|---------|------|------|
| **RAG** (Lewis et al., 2021) | 2021 | 외부 검색 인프라 | 간접적 | 중간 | 도메인 의존 |
| **Lost in the Middle** (Liu et al., 2023) | 2023 | 문제 분석 (해결책 없음) | ✗ | N/A | N/A |
| **DeepSeek-R1** (DeepSeek-AI, 2025) | 2025 | RLVR 단일 턴 추론 | ✗ | 높음 | 수학 도메인 |
| **DAPO** (Yu et al., 2025b) | 2025 | 안정적 정책 최적화 | ✗ | 높음 | 수학 도메인 |
| **MemAgent** (Yu et al., 2025a) | 2025 | 단일 문서 청크 메모리 압축 | 간접적 | 중간 | 장문 문서 |
| **Curriculum RL** (Li et al., 2026) | 2026 | 불완전 샤드 해결 가능성 탐지 | 부분적 | 낮음 | 제한적 |
| **본 논문** (Luo et al., 2026) | 2026 | 샤딩 + 롤링 메모리 RLVR | **✓ (직접적)** | **높음** | **수학+OOD** |

### 본 논문의 차별점

1. **MemAgent(Yu et al., 2025a) 대비**: MemAgent는 단일 문서 청크를 처리하지만, 본 논문은 정보가 부분적으로만 존재하는 진정한 멀티턴 대화 설정을 처음으로 RLVR로 훈련합니다.

2. **동시 연구(Li et al., 2026) 대비**: Li et al.은 불완전한 샤드 시퀀스에서의 해결 가능성 탐지에 집중하지만, 본 논문은 **완전한 시퀀스에서의 누적 신뢰성**과 **확장 가능한 데이터 구축**을 함께 다룹니다.

3. **RAG(Lewis et al., 2021) 대비**: RAG는 외부 검색 인프라를 필요로 하지만, 본 논문은 모델 자체가 메모리를 유지하는 **인프라 없는 접근법**을 제시합니다.

---

## 참고 자료

**본 논문**
- Luo, S. T., Liu, W., Liu, R., Gong, M., & Guo, J. (2026). *Multi-Turn Reasoning When Context Arrives in Pieces: Scalable Sharding and Memory-Augmented RL*. arXiv:2606.12941v2

**논문 내 인용 참고문헌**
- Laban, P., Hayashi, H., Zhou, Y., & Neville, J. (2026). *LLMs get lost in multi-turn conversation*. ICLR 2026.
- Yu, H. et al. (2025a). *MemAgent: Reshaping long-context LLM with multi-conv RL-based memory agent*.
- Yu, Q. et al. (2025b). *DAPO: An open-source LLM reinforcement learning system at scale*.
- DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning*.
- Liu, N. F. et al. (2023). *Lost in the middle: How language models use long contexts*.
- Lewis, P. et al. (2021). *Retrieval-augmented generation for knowledge-intensive NLP tasks*.
- Li, M. et al. (2026). *Mitigating lost in multi-turn conversation via curriculum RL with verifiable accuracy and abstention rewards*.
- Cobbe, K. et al. (2021). *Training verifiers to solve math word problems*. (GSM8K)
- Hendrycks, D. et al. (2021). *Measuring mathematical problem solving with the MATH dataset*. NeurIPS.
- Bai, Y. et al. (2024). *LongBench: A bilingual, multitask benchmark for long context understanding*.
- Schulman, J. et al. (2017). *Proximal policy optimization algorithms*.
- Yang, A. et al. (2024). *Qwen2.5-Math technical report*.
- Yang, A. et al. (2025). *Qwen3 technical report*.
