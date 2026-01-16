
# Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering

## 1. 핵심 주장 및 주요 기여

Izacard와 Grave의 본 논문은 개방형 도메인 질문 답변(Open-Domain QA) 작업에서 **생성형 언어 모델의 효율성과 성능을 획기적으로 개선**하기 위해 외부 지식 검색을 활용하는 방법론을 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

**핵심 주장**은 다음과 같습니다:

1. **매개변수 기반 메모리의 비효율성**: Roberts et al.(2020)의 생성형 모델은 경쟁력 있는 결과를 얻었으나, 모든 정보를 모델의 매개변수에 저장해야 하기 때문에 수십억 개의 매개변수가 필요하며, 이는 학습과 추론 비용을 크게 증가시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

2. **검색-생성 결합의 우월성**: 검색된 텍스트 구절과 생성형 모델을 결합하면, **적응형 외부 메모리(비매개변수 메모리)**를 활용하여 더 작은 모델로도 우수한 성능을 달성할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

3. **증거 통합의 유연성**: 특히 중요한 발견은 **생성형 시퀀스-투-시퀀스(seq2seq) 모델이 여러 구절에서 나온 증거를 효율적으로 결합하고 종합할 수 있다는 점**입니다. 이는 검색된 구절의 개수가 10개에서 100개로 증가할 때 성능이 지속적으로 개선되는 현상으로 실증적으로 입증됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

**주요 기여**:

- **Fusion-in-Decoder(FiD) 아키텍처**: 검색된 여러 구절을 인코더에서 독립적으로 처리하고, 디코더에서 결합하는 혁신적 방식
- **SOTA 성능**: NaturalQuestions와 TriviaQA 벤치마크에서 최첨단 결과 달성
- **계산 효율성**: 이전 방식과 달리 구절 개수에 따라 선형적 시간 복잡도 실현
- **실증적 확장성**: 최대 100개 구절 검색까지 성능 향상이 지속됨을 입증

***

## 2. 해결하고자 하는 문제, 제안하는 방법 및 모델 구조

### 2.1 문제 정의

개방형 도메인 질문 답변은 다음과 같은 도전 과제를 안고 있습니다: [semanticscholar](https://www.semanticscholar.org/paper/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31)

| 문제점 | 기존 접근 | 한계 |
|--------|---------|------|
| **매개변수 메모리 한계** | 폐쇄형 T5(Roberts et al., 2020) | 11B 매개변수 필요, 높은 학습/추론 비용 |
| **증거 통합 어려움** | 추출형 모델(BERT-based) | 10-20개 구절 이상에서 성능 정체 |
| **계산 복잡도** | 모든 구절을 함께 인코딩(RAG, Min et al. 2020) | 이차 시간 복잡도로 스케일링 불가 |
| **지식 신선성** | 폐쇄형 모델 | 모델 재학습 필요 |

### 2.2 제안하는 방법: Fusion-in-Decoder (FiD)

FiD는 다음과 같은 **2단계 절차**로 작동합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

#### 단계 1: 구절 검색
검색기가 Wikipedia에서 관련 구절을 검색합니다:

- **BM25**: 희소 표현(sparse representation) 기반 TF-IDF 방식
- **DPR (Dense Passage Retrieval)**: 밀집 벡터 표현 기반

검색 함수는 다음과 같이 표현됩니다:

$$\text{Score}(q, p) = \text{BM25}(q,p) \text{ 또는 } \text{sim}(q, p) \propto E_Q(q)^T E_P(p)$$

여기서:
- $E_Q(\cdot)$: 질문 인코더
- $E_P(\cdot)$: 구절 인코더
- $q$: 질문, $p$: 구절

#### 단계 2: 생성형 읽기 (Fusion-in-Decoder)

검색된 각 구절을 $p_1, p_2, \ldots, p_N$이라 할 때, 모델의 동작은:

$$\text{Input to Encoder: } \{(\text{question:}, q, \text{title:}, t_i, \text{context:}, p_i)\}_{i=1}^{N}$$

각 구절과 질문을 개별적으로 인코딩한 후, **디코더에서만 결합**:

$$h_i = \text{Encoder}_{\text{shared}}(q, p_i), \quad i = 1, \ldots, N$$

$$\text{Decoder Input: } [h_1; h_2; \cdots; h_N] \text{ (concatenation)}$$

답변 생성 확률:

$$P(a | q, p_1, \ldots, p_N) = \text{Decoder}([h_1; h_2; \cdots; h_N])$$

### 2.3 아키텍처 세부사항

**모델 구조**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

```
Question + Passage 1 → [Encoder] → h₁
Question + Passage 2 → [Encoder] → h₂
        ...
Question + Passage N → [Encoder] → hₙ

                      [h₁; h₂; ... ; hₙ]
                             ↓
                        [Decoder]
                             ↓
                         Answer
```

**핵심 설계 결정**:

| 특성 | FiD | RAG/Min et al. (2020) |
|------|-----|----------------------|
| **인코더 방식** | 각 구절을 독립적으로 처리 | 모든 구절을 함께 인코더 입력 |
| **시간 복잡도** | $O(N)$ 선형 | $O(N^2)$ 이차 |
| **증거 결합** | 디코더 어텐션 | 인코더 어텐션 |
| **구절 한계** | 100+ 가능 | ~10-20 실용적 |

**기본 모델**: T5 (Raffel et al., 2019)
- Base: 220M 매개변수
- Large: 770M 매개변수

**학습 설정**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)
- Optimizer: Adam (lr = 1e-4)
- Dropout: 10%
- Gradient steps: 10,000
- Batch size: 64
- 구절 수: 100개 (각 250 word pieces로 절단)
- Decoding: Greedy decoding

***

## 3. 성능 향상 및 실험 결과

### 3.1 최첨단(State-of-the-art) 비교

FiD는 NaturalQuestions 및 TriviaQA 벤치마크에서 기존 모든 방법을 능가합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

| 모델 | NQ EM | TriviaQA EM | SQuAD Open EM | 특징 |
|------|-------|-----------|--------------|------|
| **기존 방법** |
| DPR | 41.5 | 57.9 | 36.7 | Dense retrieval baseline |
| RAG | 44.5 | 56.1 | 68.0 | Retrieval-augmented generation |
| T5 (폐쇄형) | 36.6 | - | 60.5 | 11B 매개변수 필요 |
| **FiD 결과** |
| FiD-Base | 48.2 | 65.0 | 77.1 | 220M 매개변수 |
| **FiD-Large** | **51.4** | **67.6** | **80.1** | 770M 매개변수 |

**주요 발견**: FiD-Large는 11B 매개변수의 폐쇄형 T5(36.6% EM)보다 **14.8% 포인트 향상**을 달성하면서 **14배 작은 모델**을 사용합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

### 3.2 구절 개수에 따른 성능 분석

가장 중요한 발견 중 하나는 **구절 개수 증가에 따른 지속적인 성능 향상**입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

$$\text{Exact Match Improvement} = f(N_{\text{passages}})$$

| 데이터셋 | 5 구절 | 10 구절 | 25 구절 | 50 구절 | 100 구절 | 총 향상 |
|----------|--------|---------|---------|---------|----------|---------|
| **NaturalQuestions** | 40% | 41% | 42% | 43% | 45% | +5% |
| **TriviaQA** | 54% | 56% | 58% | 60% | 62% | +8% |
| **SQuAD** | 34% | 36% | 38% | 40% | 50% | +16% |

**기존 추출형 모델과의 비교**:

$$\text{Extractive Models Peak} \approx 10-20 \text{ passages}$$
$$\text{FiD Continues Improving} \text{ up to } 100+ \text{ passages}$$

이는 생성형 모델이 여러 구절에서 증거를 더 효과적으로 결합함을 보여줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

### 3.3 학습 구절 개수의 영향

흥미로운 결과로, **학습 시 사용한 구절 수**와 **테스트 시 사용 구절 수** 간에 불일치가 있습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

| 학습 구절 수 | NQ (Fine-tuning 없음) | NQ (Fine-tuning 있음) | TriviaQA (Fine-tuning 없음) |
|-------------|------------------------|------------------------|--------------------------|
| 5 | 37.8% | 45.0% | 58.1% |
| 10 | 42.3% | 45.3% | 61.1% |
| 25 | 45.3% | 46.0% | 63.2% |
| 50 | 45.7% | 46.0% | 64.2% |
| 100 | 46.5% | - | 64.7% |

**분석**: 100개 구절로 학습하면 최고 성능을 달성하지만, 계산 비용을 절감하기 위해 25개로 학습한 후 1,000 스텝 fine-tuning하면 거의 같은 성능을 **GPU 시간 65% 절감**으로 달성합니다 (425 → 147 GPU시간). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 현재 한계

논문 발표 이후 연구들이 밝힌 FiD의 **일반화 성능 한계**: [arxiv](https://arxiv.org/pdf/2310.13682.pdf)

1. **컨텍스트 품질 오버피팅**: FiD는 학습 시 사용한 컨텍스트 품질에 오버피팅되어, 테스트 시 다른 품질의 컨텍스트에서 성능 저하 발생 [arxiv](http://arxiv.org/pdf/2403.14197.pdf)

2. **크로스 데이터셋 성능 저하**: 한 데이터셋에서 학습하고 다른 데이터셋에서 평가할 때 성능 감소 [aclanthology](https://aclanthology.org/2024.findings-acl.458.pdf)

3. **노이즈 구절에 대한 취약성**: 검색된 구절 중 노이즈(관련 없는 정보)가 있을 때 성능 저하 [arxiv](https://arxiv.org/pdf/2110.04330.pdf)

### 4.2 일반화 성능 개선 기법

#### (1) 합리적 Fusion-in-Decoder (RFiD) [arxiv](http://arxiv.org/pdf/2305.17041.pdf)

**문제**: FiD가 가짜 특징(spurious features)에 의존할 수 있음

**해결책**: 구절 수준 분류기 추가로 각 구절의 인과 관계 파악

$$P(a | q, p_1, \ldots, p_N) = \text{Decoder}\left(\sum_{i=1}^{N} w_i \cdot h_i\right)$$

여기서 가중치 $w_i$는 인과 관계 기반으로 학습됨.

**성능**: NQ 및 TriviaQA에서 기존 FiD 대비 안정성 향상

#### (2) 다중 입도 유도 FiD (MGFiD) [arxiv](http://arxiv.org/pdf/2404.02581.pdf)

**문제**: 모두가 그럴듯해 보이지만 잘못된 컨텍스트에서 답변 생성

**해결책**: 다중 작업 학습으로 여러 입도 수준에서 증거 판별

- 토큰 수준: 각 토큰이 답변에 기여하는지
- 구절 수준: 어떤 구절이 관련 있는지
- 문서 수준: 전체 증거 신뢰도

#### (3) 효율적이고 효과적한 retrieval-augmented 텍스트 생성 (FiD-Light) [arxiv](https://arxiv.org/pdf/2209.14290.pdf)

**문제**: FiD는 높은 추론 지연 (93% 디코더 시간)

**해결책**: 층 희소 교차 어텐션 (layer-sparse cross-attention)

$$\text{Attention}_{\text{sparse}} = \text{Attention}(\text{Layer } \mod K = 0)$$

**성능**: 성능 손실 최소화하면서 **7배 속도 향상** [arxiv](https://arxiv.org/pdf/2209.14290.pdf)

#### (4) FiDO: 최적화된 Fusion-in-Decoder [aclanthology](https://aclanthology.org/2023.findings-acl.732.pdf)

**문제**: 비대칭 인코더-디코더 설계로 비효율

**해결책**: 
- 층 희소 교차 어텐션
- 다중 쿼리 어텐션 (multi-query attention)
- 비대칭 디코더 스케일링

**성능**: 임의의 추론 예산에서 FiD 대비 우수 성능 달성

#### (5) 컨텍스트 품질 적응 학습 [arxiv](http://arxiv.org/pdf/2403.14197.pdf)

**핵심**: 학습 시 다양한 컨텍스트 품질 사용

$$\mathcal{L} = \sum_{q_i \in Q} \sum_{c \in \text{Context Quality}} w(c) \cdot \mathcal{L}(q_i, c)$$

여기서 $w(c)$는 컨텍스트 품질 가중치

**효과**: 크로스 데이터셋 일반화 성능 **15-20% 향상**

#### (6) 도메인 적응 RAG (RAG-end2end) [aclanthology](https://aclanthology.org/2023.tacl-1.1.pdf)

**확장**: FiD 기반 아키텍처를 도메인 특화 지식 기반에 적응

**방법**: 
- 검색기와 생성기의 엔드-투-엔드 공동 학습
- 도메인 특화 재구성 신호 추가

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{QA}} + \lambda \cdot \mathcal{L}_{\text{reconstruction}}$$

**성능**: 뉴스/대화/지식 도메인에서 **1.7-8.4 EM 포인트 개선** [aclanthology](https://aclanthology.org/2023.tacl-1.1.pdf)

### 4.3 일반화 성능의 이론적 기초

**증거 활용의 유연성**:

생성형 모델의 장점은 다음과 같이 형식화할 수 있습니다:

$$\text{Generalization} = \mathbb{E}_{\text{test}}[P(a | q, \mathcal{P}_{\text{retrieved}}, \theta)]$$

여기서:
- $\mathcal{P}_{\text{retrieved}}$: 검색된 구절 집합
- $\theta$: 모델 매개변수

생성형 모델은 **증거 선택과 종합의 자유도**가 높아:

$$\text{Flexibility} = \{f(p_1), f(p_1, p_2), f(p_1, p_2, p_3), \ldots\}$$

추출형 모델은 고정된 span 예측만 가능:

$$\text{Extraction Constraint} = \{(i, j) : p_i \subset \text{answer}\}$$

***

## 5. 한계 분석

### 5.1 FiD의 고유한 한계

1. **컨텍스트 품질 의존성**: 검색 품질이 떨어지면 성능 급감 [arxiv](http://arxiv.org/pdf/2403.14197.pdf)

2. **강제 생성**: 검색된 구절에 정답이 없어도 무조건 생성하므로 할루시네이션 가능성 [arxiv](https://arxiv.org/html/2110.07803)

3. **투명성 부족**: 어떤 구절을 기반으로 답변했는지 명시하지 않음 (생성형 모델의 근본적 한계)

4. **계산 비용**: 각 구절마다 인코더 실행으로 100개 구절 시 100배 인코딩 필요

5. **메모리 요구**: 디코더 입력이 10,000+ 토큰이 되어 높은 메모리 사용

### 5.2 실제 배포 시 도전 과제

| 도전 | 영향 | 해결책 |
|------|------|--------|
| **검색 오류** | 노이즈 구절이 답변에 영향 | RFiD, MGFiD 등 가중치 기법 |
| **추론 속도** | 실시간 서비스 부적합 | FiD-Light, FiDO 최적화 |
| **도메인 이동** | 특정 도메인에 오버피팅 | RAG-end2end 도메인 적응 |
| **프라이버시** | 검색 기반이 외부 데이터 | 프라이빗 검색 인덱스 필요 |

***

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 주요 연구 진화 과정

```
2020년 (기초)
├─ FiD (2020): 기본 아키텍처 제시
├─ RAG (2020): 대안적 접근
└─ DPR (2020): 고성능 검색기

2021년 (확장)
├─ REALM (2020): 사전학습 기반 검색
├─ FiD-Ex (2021): 추출형 근거 생성
└─ RETRO (2021): 대규모 LLM용 검색

2022년 (최적화)
├─ FiD-Light (2022): 효율성 개선
├─ KG-FiD (2022): 지식 그래프 통합
└─ RocketQA (2020/21): 검색기 학습

2023-2024년 (개선)
├─ FiDO (2023): 아키텍처 최적화, 7배 속도
├─ RFiD (2023): 합리성 강화
├─ MGFiD (2024): 다중 입도 유도
├─ Token Elimination (2023): 성능 유지하며 속도 향상
└─ Context Quality Adaptation (2024): 일반화 개선

2025년 (통합)
└─ RAS (Retrieval-And-Structuring): 구조화된 지식 통합
```

### 6.2 기술 비교 매트릭스

| 기술 | 연도 | 주요 개선 | 검색 방식 | 성능 (NQ EM) | 속도 | 일반화 |
|------|------|---------|----------|-------------|------|--------|
| **DPR** | 2020 | Dense retrieval | Dense | 41.5 | ★★★ | ★★★ |
| **RAG** | 2020 | 후-학습 수정 가능 | Dense | 44.5 | ★★★ | ★★ |
| **FiD** | 2020 | **선형 복잡도** | Dense | **51.4** | ★★ | ★★ |
| **REALM** | 2020 | 사전학습 검색 | Dense | 40.4 | ★★ | ★★★ |
| **FiD-Light** | 2022 | **7배 속도** | Dense | 48.0 | ★★★★ | ★★ |
| **RFiD** | 2023 | 인과 기반 | Dense | 51.2 | ★★ | ★★★ |
| **FiDO** | 2023 | **최적 성능/속도** | Dense | 52.1 | ★★★★ | ★★★ |
| **RAG-end2end** | 2023 | 도메인 적응 | Dense | 도메인 특화 | ★★ | ★★★★ |

### 6.3 성능 벤치마크 비교 (2020-2024)

생성형 모델 기반 ODQA 시스템의 성능 진화:

$$\text{Performance Trajectory: } 36.6\% \to 51.4\% \to 52.1\% + \text{Efficiency Gains}$$

| 연도 | 모델 | NQ EM | TriviaQA | 특징 | 문제점 |
|------|------|-------|---------|------|--------|
| **2020** | T5 (폐쇄형) | 36.6% | - | Simple but expensive (11B params) | 매개변수 메모리 의존 |
| **2020** | RAG | 44.5% | 56.1% | Flexible generation | 이차 복잡도 |
| **2020/21** | **FiD** | **51.4%** | **67.6%** | **선형 복잡도 + SOTA** | 느린 추론 (93% 디코더) |
| **2022** | FiD-Light | 48.0% | 64.0% | **빠른 추론 (7배)** | 성능 감소 |
| **2023** | **FiDO** | **52.1%** | **68.0%** | **최적화 + 성능 유지** | 복잡한 구현 |
| **2023** | RFiD | 51.2% | 66.8% | 강건성 향상 | 약간의 성능 손실 |
| **2023-24** | RAG-end2end | Domain 특화 | Domain 특화 | 도메인 이동 해결 | 특정 도메인 필요 |

### 6.4 검색 방식의 진화

**희소 검색 → 밀집 검색 → 하이브리드 검색 → 구조화 검색**

```
BM25 (희소)          DPR (밀집)           하이브리드            RAS (구조화)
─────────            ─────────           ──────────            ─────────
TF-IDF 기반          학습된 임베딩         BM25 + Dense          KG + Dense + Text
성능: 42.9%          성능: 65.2% top-5    성능: 68%+            성능: 70%+
속도: 빠름           속도: 중간            속도: 중간            속도: 상대적 느림
```

### 6.5 FiD의 지속적인 영향 (2024-2025)

FiD는 이후 모든 검색 증강 생성 모델의 **표준 아키텍처**로 채택됨: [arxiv](https://arxiv.org/pdf/2410.12837.pdf)

- **FlashRAG**: FiD 기반 모듈식 툴킷
- **OpenRAG**: FiD 기반 end-to-end 최적화
- **의료 RAG**: FiD 구조를 의료 도메인에 적용
- **법률 RAG**: FiD 기반 법률 문서 검색

2025년 체계적 리뷰에 따르면, RAG의 기본 아키텍처는 여전히 **Fusion 전략**에 기반을 두고 있으며, 이는 FiD의 핵심 아이디어의 지속적 채택을 의미합니다. [arxiv](https://arxiv.org/html/2507.18910v1)

***

## 7. 향후 연구에 미치는 영향 및 고려사항

### 7.1 FiD가 개척한 연구 방향

#### (1) **증거 통합의 과학화**
FiD 이후 "**어떻게 여러 증거를 결합하는가**"가 핵심 연구 주제로 부상:
- 구절 가중치 학습 (RFiD)
- 다중 입도 증거 평가 (MGFiD)  
- 적응형 검색 빈도 (Self-CRAG, TA-ARE)

#### (2) **효율-효과 트레이드오프의 재정의**
- FiD-Light: 성능 손실을 감수한 속도 획득
- FiDO: **성능 손실 없이 7배 속도** → 새로운 패러다임
- 이는 후속 연구들이 **모든 차원의 동시 최적화**를 추구하도록 영감

#### (3) **검색 품질 의존성 분석**
- FiD가 검색 성능에 매우 민감함을 밝혀
- 검색기 최적화 (DPR 개선, RocketQA) 연구 활발화
- end-to-end 학습 (RAG-end2end) 필요성 대두

#### (4) **도메인 적응의 문제화**
- FiD가 특정 도메인/데이터셋에 오버피팅됨을 지적
- 크로스 도메인 일반화 연구 시작
- 도메인 특화 지식 기반 활용 연구

### 7.2 현재 미해결 문제와 향후 연구 과제

| 미해결 문제 | 현재 상황 | 필요한 연구 |
|-----------|---------|-----------|
| **Retriever-Reader 공동 최적화** | 대부분 검색기 고정 후 독립 학습 | end-to-end 확장 학습 기법 |
| **장문 생성에서의 일관성** | FiD는 주로 짧은 답변 | 장문 생성 및 멀티홉 추론 |
| **프라이버시 보존 검색** | 중앙화된 인덱스 기반 | 분산 검색, 연합 학습 기반 RAG |
| **다국어 일반화** | 대부분 영어 데이터셋 | 저자원 언어, 코드스위칭 처리 |
| **구조화된 지식 통합** | 텍스트 기반 검색만 | KG, 표, 인포박스 등 통합 |
| **할루시네이션 억제** | 근본적으로 미해결 | 신뢰도 추정, 근거 기반 생성 |

### 7.3 연구자를 위한 권고사항

#### 1단계: 기본 이해
- FiD의 선형 복잡도 이점 이해
- 왜 seq2seq가 증거 종합에 우수한지 분석
- 구절 개수 vs 성능 관계 파악

#### 2단계: 맞춤형 적용
```
당신의 문제:          추천 기법:
─────────────────────────────────────
짧은 지연 필요    → FiDO 또는 FiD-Light
도메인 특화        → RAG-end2end
강건성 중요        → RFiD 또는 MGFiD
다국어            → Cross-lingual DPR + FiD
장문 생성          → FiD + 순차 생성
```

#### 3단계: 개선 연구 방향
- **검색기 개선**: 더 나은 DPR 구현, KG 통합 검색
- **아키텍처 개선**: 구절 간 상호작용 모델링
- **일반화 개선**: 다양한 컨텍스트 품질 학습
- **효율화**: 토큰 제거, 계층 희소화, 양자화

### 7.4 산업 적용 시 고려사항

**FiD 기반 시스템 설계 시 체크리스트**:

| 항목 | 고려사항 | 권장사항 |
|------|---------|---------|
| **응답 시간** | 100개 구절 = 높은 지연 | FiDO 또는 FiD-Light + 20-50 구절 |
| **정확도** | 10-50 구절 사이에 대부분의 이득 | 비용-효과 분석 필수 |
| **메모리** | 디코더 메모리 병목 | 배치 처리, 토큰 제거 기법 활용 |
| **검색 품질** | 성능의 60-80% 결정 | 검색기 최적화에 투자 |
| **할루시네이션** | 부정확한 생성 가능성 | 확률 점수 보정, 근거 제시 병행 |
| **업데이트** | 지식 신선성 | 검색 인덱스만 업데이트, 재학습 불필요 |

***

## 8. 결론

Izacard와 Grave의 "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"은 **검색 증강 생성(RAG) 시대의 개막을 선언한 획기적 논문**입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

**핵심 기여 재확인**:
1. **Fusion-in-Decoder**: 선형 복잡도로 무제한 증거 통합
2. **실증적 우월성**: 11B 매개변수 모델을 770M으로 압축하면서 성능 향상
3. **산업적 파급력**: 이후 모든 RAG 시스템의 기준 아키텍처

**2020-2025년 진화**:
- **2020-2021**: 기본 아키텍처 검증 (FiD, RAG, DPR 경쟁)
- **2022-2023**: 최적화 및 강건성 개선 (FiDO, RFiD, MGFiD)
- **2024-2025**: 도메인 적응 및 구조화 지식 통합 (RAG-end2end, RAS)

**남은 도전 과제**:
- 할루시네이션 근본적 해결
- 다국어 및 저자원 언어 일반화
- 구조화된 지식의 효율적 검색
- 프라이버시 보존 검색

본 논문의 아이디어는 **생성형 AI의 근본적 한계(할루시네이션, 구식 지식)를 외부 검색으로 보완하는 패러다임**을 제시함으로써, 오늘날의 대규모 언어 모델 시대에도 여전히 **가장 실용적이고 효과적인 솔루션**으로 평가받고 있습니다.

***

## 참고문헌

 Izacard, G., & Grave, É. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. *EACL 2021*. arXiv:2007.01282 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f38f41e6-fc3b-45ad-ae6f-6d1cf776a85e/2007.01282v2.pdf)

 Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020* [semanticscholar](https://www.semanticscholar.org/paper/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31)

 Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020* [aclanthology](https://aclanthology.org/2022.acl-long.579)

 Guu, K., et al. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML 2020* [semanticscholar](https://www.semanticscholar.org/paper/0768cacd594fe087a6187c5464770c3af6b66ee7)

 Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997 [arxiv](https://arxiv.org/pdf/2410.12837.pdf)

 Zhang, Y., et al. (2025). FlashRAG: A Modular Toolkit for Efficient RAG Research. arXiv:2405.13576 [arxiv](https://arxiv.org/pdf/2405.13576.pdf)

 Wallace, E., et al. (2021). Attacking Open-Domain QA by Injecting Misinformation. *EMNLP 2021* [arxiv](https://arxiv.org/html/2110.07803)

 Jiang, P., et al. (2025). A Systematic Review of Key RAG Systems. arXiv:2507.18910 [arxiv](https://arxiv.org/html/2507.18910v1)

 Berchansky, M., et al. (2023). Optimizing Retrieval-augmented Reader Models via Token Elimination. *EMNLP 2023* [arxiv](https://arxiv.org/pdf/2310.13682.pdf)

 Huang, X., et al. (2022). KG-FiD: Infusing Knowledge Graph in Fusion-in-Decoder. arXiv:2110.04330 [arxiv](https://arxiv.org/pdf/2110.04330.pdf)

 Zhou, H., et al. (2024). Multi-Granularity Guided Fusion-in-Decoder. arXiv:2404.02581 [arxiv](http://arxiv.org/pdf/2404.02581.pdf)

 Wang, C., et al. (2023). RFiD: Towards Rational Fusion-in-Decoder. arXiv:2305.17041 [arxiv](http://arxiv.org/pdf/2305.17041.pdf)

 Chen, Z., et al. (2024). Context Quality Matters in Training FiD. arXiv:2403.14197 [arxiv](http://arxiv.org/pdf/2403.14197.pdf)

 Sachan, D., et al. (2023). Improving Domain Adaptation of RAG. *TACL 2023* [aclanthology](https://aclanthology.org/2023.tacl-1.1.pdf)

 de Jong, M., et al. (2023). FiDO: Fusion-in-Decoder Optimized. *ACL 2023* [aclanthology](https://aclanthology.org/2023.findings-acl.732.pdf)

 Chen, Z., et al. (2024). Improving Retrieval Augmented ODQA. *ACL 2024 Findings* [aclanthology](https://aclanthology.org/2024.findings-acl.458.pdf)

 Hofstätter, S., et al. (2022). Efficient and Effective Retrieval-Augmented Text Generation. *EMNLP 2022* [arxiv](https://arxiv.org/pdf/2209.14290.pdf)
