# Corrective Retrieval Augmented Generation 

## 1. 논문의 핵심 주장과 주요 기여
### 1.1 해결하고자 하는 문제
**CRAG**는 Retrieval-Augmented Generation(RAG) 시스템에서 근본적인 취약점을 다룬다. 대규모 언어모델(LLM)은 파라미터 지식만으로 생성하면서 환각(hallucination) 현상을 겪는다. RAG는 외부 문서 검색을 통해 이를 보완하지만, 검색 결과의 질에 과도하게 의존한다는 치명적 한계가 있다. 논문은 세 가지 근본 문제를 지적한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

1. **검색 실패에 대한 대응 부재**: 기존 RAG는 검색이 실패해도 검색된 부적절한 문서를 무조건 사용
2. **문서 전체 활용의 비효율성**: 검색된 전체 문서를 동등하게 취급하나, 실제로는 부분 정보만 필요
3. **검색 결과 품질 미분화**: 높은 관련성과 낮은 관련성의 문서를 구분하지 않음

### 1.2 세 가지 주요 기여
논문의 기여는 다음 세 가지로 집약된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

1. **검색 실패 시나리오의 첫 번째 대응**: "검색이 실패했을 때 어떻게 하는가"라는 질문에 처음 정면으로 대응. 이는 기존 RAG 개선 논문들(Self-RAG, RAFT)과 근본적으로 다른 관점
2. **플러그-앤-플레이 방법론**: 경량 검색 평가기와 웹 검색 통합으로 기존 RAG 시스템에 빠르게 적용 가능
3. **광범위한 일반화 능력**: 4개 이상의 이질적 데이터셋(단문 생성, 장문 생성, 참/거짓 판정, 다중선택)에서 일관된 성능 향상 입증

## 2. 제안 방법의 상세 설명
### 2.1 핵심 수식 및 개념
RAG 프레임워크는 다음 확률 분해로 표현된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

$$P(Y|X) = P(D|X)P(Y|D|X)$$

여기서:
- $X$: 입력 질문
- $D$: 검색된 문서들
- $Y$: 생성된 응답

이 식은 **검색기(Retriever)와 생성기(Generator)의 긴밀한 결합**을 보여주며, 검색 실패 시 생성기의 능력이 아무리 뛰어나도 만족스러운 응답 생성이 불가능함을 의미한다.

### 2.2 모델 구조: CRAG의 네 가지 핵심 컴포넌트
#### A. 검색 평가기 (Retrieval Evaluator)

**아키텍처**: T5-large (0.77B) 기반 미세조정 모델

**점수 계산 방식**:
- 각 (질문, 문서) 쌍에 대해 독립적으로 관련성 점수 계산
- 점수 범위: -1 (완전 부관련) ~ +1 (완벽한 관련)
- ChatGPT 기반 평가(58.0-64.7%)대비 84.3% 정확도 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

**학습 데이터**:

$$\text{Positive samples: } \{d \in D | \text{title}(d) = \text{golden wiki title}(x)\}$$

$$\text{Negative samples: } \{d' \in D | \text{title}(d') \neq \text{golden wiki title}(x)\}$$

**선택 근거**: 경량성으로 실시간 평가 가능하며, 도메인 간 전이 학습에 유리함.

#### B. 행동 트리거 (Action Trigger) - 3단계 신뢰도 결정 메커니즘

평가기의 점수 분포를 세 개 신뢰도 등급으로 변환:

**1) Correct 행동** (신뢰도 상한 초과)
- 조건: $\max_{i} \text{score}_i > \text{threshold high}$
- 의미: 검색된 문서 중 충분히 관련성 높은 문서 존재
- 후속 처리: 문서 정제(Knowledge Refinement)

**2) Incorrect 행동** (신뢰도 하한 미만)
- 조건: $\max_{i} \text{score}_i < \text{threshold low}$
- 의미: 모든 검색 문서가 부적절
- 후속 처리: 웹 검색(Web Search) 수행

**3) Ambiguous 행동** (중간 범위)
- 조건: $\text{threshold low} \leq \max_{i} \text{score}_i \leq \text{threshold high}$
- 의미: 평가기가 판정에 불확실
- 후속 처리: 내부 지식 + 외부 지식 모두 활용

**임계값 설정** (경험적 결정): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)
| 데이터셋 | Upper | Lower |
|---------|-------|-------|
| PopQA | 0.59 | -0.99 |
| PubHealth, Arc | 0.50 | -0.91 |
| Biography | 0.95 | -0.91 |

**디자인 철학**: Ambiguous 행동은 평가기 오류에 대한 견고성을 크게 향상시킨다. 이를 제거했을 때 성능 저하가 -0.8~-0.9% 정도로 작지만, 기존 Correct/Incorrect만으로 학습했을 때 성능 변동성이 훨씬 크다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

#### C. 지식 정제: 분해-필터-재구성 알고리즘

**목표**: 검색된 문서에서 생성에 필요한 핵심 정보만 추출

**단계별 처리**:

1) **분해 (Decomposition)**
   - 문서를 의미론적 단위(strip)로 분할
   - 규칙: 1~2문장 → 개별 strip, 이상 → 자동 분할
   - 목적: 문서 내 노이즈 제거

2) **필터링 (Filtering)**
   $$k_{\text{internal}} = \{\text{strip}_j | E(x, \text{strip}_j) > -0.5, \text{ rank}_j \leq 5\}$$
   - 각 strip에 검색 평가기 재적용
   - 상위-5 관련도 strip 선택
   - 필터 임계값: -0.5

3) **재구성 (Recomposition)**

$$k_{\text{final}} = \text{concatenate}(\text{sorted strips})$$
   
   - 선택된 strip을 원본 순서대로 연결

**효과**: 절제 연구 결과, 문서 정제 제거 시 성능 저하 5.1~9.6% 발생 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

#### D. 웹 검색 (Web Search)

**프로세스**:

1) **질의 재작성**:
   $$q_{\text{search}} = \text{ChatGPT}(\text{prompt: "Extract keywords from Q"})$$
   - 3개 이하의 키워드 추출
   - 예시: "Death of a Batman" + "screenwriter" → 웹 검색

2) **검색 실행**:
   - Google Search API 활용
   - Top-5 URL 반환
   - Wikipedia 페이지 우선순위 지정

3) **정제**:
   - 내부 지식과 동일한 분해-필터-재구성 알고리즘 적용

$$k_{\text{external}} = \text{Knowledge Refine}(x, \text{web content})$$

**의도**: 정적 코퍼스의 한계(범위, 다양성, 시간 최신성)를 동적 웹 정보로 보완

### 2.3 추론 알고리즘 (Algorithm 1)
```
함수 CRAG_Inference(x, D, E, W, G):
    입력: 질문 x, 검색 문서 D, 평가기 E, 질의 재작성기 W, 생성기 G
    
    // 단계 1: 검색 평가
    for each d_i in D:
        score_i ← E.evaluate(x, d_i)  // 관련성 점수 계산
    
    // 단계 2: 신뢰도 판정
    confidence ← aggregate({score_1, ..., score_k})  
    // confidence ∈ {CORRECT, INCORRECT, AMBIGUOUS}
    
    // 단계 3: 행동 선택 및 지식 구성
    if confidence == CORRECT:
        k ← Knowledge_Refine(x, D)
    else if confidence == INCORRECT:
        q_search ← W(x)
        k ← Web_Search(q_search)
    else:  // AMBIGUOUS
        k_in ← Knowledge_Refine(x, D)
        q_search ← W(x)
        k_ex ← Web_Search(q_search)
        k ← k_in + k_ex
    
    // 단계 4: 생성
    y ← G(x, k)
    return y
```

이 알고리즘의 **핵심 특징**:
- **적응성**: 각 질문에 따라 동적으로 행동 선택
- **보수성**: 신뢰도가 낮을 때 여러 지식원 활용
- **효율성**: 검색 재평가 1회 추가만으로 다양한 시나리오 처리

## 3. 성능 향상의 정량적 분석
### 3.1 주요 성능 지표
**PopQA (단문 생성)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)
- RAG → CRAG: 50.5% → 54.9% (+4.4%)
- RAG → Self-RAG: 50.5% → 54.9% (동등)
- CRAG의 이점: Self-RAG 동등의 성능을 비교 대상이 아닌 선택적 적용으로 달성

**Biography (장문 생성)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)
- RAG → CRAG: 44.9 FactScore → 47.7 (+2.8)
- 이는 생성된 각 원자적 사실의 정확도를 평가한 메트릭
- 장문 생성에서 CRAG의 효과가 상대적으로 낮은 이유: 이미 Self-RAG(81.2)가 높은 성능 달성

**PubHealth (참/거짓 판정)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)
- RAG → CRAG: 48.9% → 59.5% (+10.6%) ← **최대 개선폭**
- 이유: 의료 도메인에서 정확한 관련성 판정이 중요하며, 잘못된 정보 제공 위험이 높음

**Arc-Challenge (다중선택)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)
- RAG → CRAG: 43.4% → 53.7% (+10.3%)
- CRAG → Self-CRAG: 53.7% → 67.2% (+13.5%)

### 3.2 모델 일반화 능력의 실증 분석
**문제 상황**: Self-RAG는 특정 LLM(SelfRAG-LLaMA2-7b)의 reflection token 생성 능력에 의존. 다른 LLM으로 변경 시 이 능력이 없으므로 성능 급락.

**CRAG의 우월성**: 평가기가 LLM과 독립적이므로 임의의 생성 LLM과 결합 가능: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

| 데이터셋 | Self-RAG 성능 차이 | Self-CRAG 성능 차이 |
|---------|------------------|------------------|
| PopQA | -25.9% | -12.8% |
| Biography | -48.9% | -17.1% |
| PubHealth | -71.7% | -74.2% |

**분석**: 
- PubHealth의 경우 Self-CRAG도 성능이 떨어지는 이유: LLaMA2-hf-7b 자체의 이진 분류 지시문 이해 능력 부족
- 그럼에도 Self-RAG의 -71.7% 대비 Self-CRAG의 -74.2%는 CRAG 자체의 한계가 아니라 기본 생성기의 한계를 반영

### 3.3 검색 품질 저하에 대한 견고성
**실험 설계**: PopQA에서 정확한 검색 결과의 일부를 임의로 제거하여 검색 품질 저하 시뮬레이션 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

**결과 해석**:
- 검색 정확도 69.8% → 20%로 감소 시:
  - Self-RAG: 62.4% → 28.1% (34.3 포인트 하락)
  - Self-CRAG: 63.2% → 38.2% (25 포인트 하락)

**핵심 통찰**: CRAG의 웹 검색 메커니즘이 검색 실패 상황에서 작동하여 성능 저하를 완화. 특히 검색 정확도 30~40% 대역에서 CRAG의 상대적 우위가 최대.

### 3.4 절제 연구(Ablation Study)
**행동별 영향도** (PopQA, 제거 시 성능 저하): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

$$\Delta \text{Performance}_{\text{Correct}} = -1.7\%$$

$$\Delta \text{Performance}_{\text{Incorrect}} = -0.5\%$$

$$\Delta \text{Performance}_{\text{Ambiguous}} = -0.9\%$$

**해석**:
- Correct 행동의 영향이 가장 크다 (내부 지식이 대부분의 경우 필요)
- Incorrect 행동의 영향이 작지만 중요 (일부 질문에서만 활성화)
- Ambiguous 행동이 평가기 오류에 대한 buffer 역할

**지식 활용 연산별 영향도**:
- 문서 정제 제거: -5.1% ~ -9.6% ← **가장 큰 영향**
- 질의 재작성 제거: -3.2% ~ -3.6%
- 선택 제거: -4.0% ~ -1.2%

## 4. 일반화 성능 향상의 메커니즘
### 4.1 왜 CRAG가 일반화하는가?
**세 가지 구조적 이점**:

1. **경량 평가기의 도메인 적응성**:
   - T5-large는 특정 작업에 overfitting되지 않음
   - PopQA에서 미세조정된 평가기가 Biography, PubHealth, Arc-Challenge에서도 효과적
   - 이는 관련성 판정이 작업 독립적 특성임을 시사

2. **휴리스틱 기반 의사결정**:
   - 행동 트리거가 고정 규칙 기반 (상한값 초과 → Correct)
   - 특정 작업 형식에 특화되지 않음
   - 단문, 장문, 참/거짓, 다중선택 등 모두에 적용 가능

3. **모듈식 아키텍처**:
   - 평가기, 정제 알고리즘, 웹 검색이 독립적 모듈
   - 각 컴포넌트를 독립적으로 개선 가능
   - Self-RAG처럼 생성기 구조 변경 불필요

### 4.2 일반화 한계 및 개선 방향
**명시적 한계** (논문에서 인정): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)
- 새로운 도메인에서도 평가기 미세조정 필요
- 데이터셋별로 임계값 수동 튜닝 필요 (PopQA: 0.59 vs Biography: 0.95)

**암묵적 한계**:
- **Zero-shot 일반화 미검증**: 다른 도메인에 완전히 미세조정 없이 적용 시 성능 미실증
- **검색 소스 의존성**: 웹 검색 API에 의존하므로 폐쇄 환경에서 제한적
- **계산 오버헤드**: 평가기 추론 추가로 Table 6에 따르면 27.2 TFLOPs/token (RAG 26.5 대비 2.7% 증가)

## 5. 2020년 이후 관련 최신 연구 비교 분석
### 5.1 핵심 선행 연구 맵
| 방법 | 발표 | 핵심 아이디어 | CRAG와의 차이 |
|------|------|------------|-----------|
| **RAG** | 2020 | 검색 + 생성 파이프라인 | 고정 K개 문서 무조건 사용 |
| **Self-RAG** | 2024 (ICLR) | 반사 토큰으로 선택적 검색 | LLM instruction-tuning 필수 |
| **RAFT** | 2024 (COLM) | 도메인 특화 미세조정 | 검색 실패 대응 없음 |
| **CRAG** | 2024 | 검색 평가 + 웹 보완 | **검색 실패 직접 대응** |

### 5.2 CRAG의 차별화된 포지셔닝
**Self-RAG와의 비교**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)
- Self-RAG: "언제 검색해야 하는가?"에 집중
- CRAG: "검색이 실패했을 때 어떻게 할 것인가?"에 집중
- 결과: 보완적 접근 → Self-CRAG 결합으로 최고 성능 달성

**RAFT와의 비교**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)
- RAFT: 도메인 특화 문서 집합에 최적화
- CRAG: 도메인 불특정 일반적 방법
- RAFT 성능: PubMed에서 35.25% 개선
- CRAG 성능: PubHealth에서 36.6% 개선 (비교 가능한 수준)

### 5.3 2024-2025 최신 연구 동향
**1. Auto-RAG (2024)**
- 특징: 반복 검색 시 LLM의 추론 능력 활용
- CRAG와의 관계: CRAG의 웹 검색을 더 지능화한 방향
- 차이: CRAG는 단순 키워드 기반, Auto-RAG는 LLM 추론 기반

**2. Chain-of-Retrieval RAG (2025)**
- 특징: o1 유사 단계별 chain-of-thought 검색과 추론
- CRAG와의 관계: CRAG는 1회 평가로 행동 결정, 이 방법은 다단계 추론
- 성능: CRAG보다 높을 것으로 예상되나 계산 비용 증가

**3. RARE 평가 프레임워크 (2025)**
- 특징: RAG 시스템의 견고성 체계적 평가
- CRAG와의 관계: CRAG는 실제 개선 방법, RARE는 평가 방법
- 의의: RARE로 평가했을 때 CRAG의 견고성 우월성 재확인 가능

**4. RAG 견고성 연구 (2024-2025)**
- 발견: 역대적 문서, 쿼리 프레이밍 변화에 취약
- CRAG의 역할: 이러한 견고성 문제의 실제 해결책 제시
- 영향: CRAG 기법이 표준 견고성 강화 방법으로 인정될 가능성

### 5.4 CRAG의 학술적 위치
- **시간적 위치**: 2024년 초 (비교적 최신, 2025년 초 기준 약 1년 된 논문)
- **개념적 위치**: Self-RAG 이후의 고도화된 개선 → "검색 평가" 측면 강화
- **영향 범위**: Self-RAG의 "적응형 검색" 개념과 보완적으로 작동
- **미래 전망**: 
  - 적응형 임계값 학습으로 한계 극복 가능
  - LLM 자체에 평가 기능 내재화로 경량화 가능
  - 멀티홉 추론 RAG로 확장 가능

## 6. 향후 연구 시 고려할 핵심 사항
### 6.1 이론적 개선 방향
**1. 적응형 신뢰도 계산**
- 현재: 데이터셋별 고정 임계값 (PopQA: 0.59, Biography: 0.95)
- 제안: 메타 학습으로 쿼리별 최적 임계값 동적 결정

$$\text{threshold}\_{\text{adaptive}} = f_{\theta}(x, D) \quad \text{(학습 가능)}$$

**2. 다원 지식 소스 통합**
- 현재: 문서 검색 vs 웹 검색 이원선택
- 제안: 지식 그래프, API, 캐시, 실시간 데이터 등 다중 소스
- 이점: 특정 도메인(금융, 과학)에서 정확도 향상

**3. 설명 가능성 강화**
- 현재: "평가기가 점수 0.65를 부여함" → 블랙박스
- 제안: 평가 근거 생성 (어떤 단어/개념이 관련성 판정에 영향)
- 이점: 사용자 신뢰도 향상, 디버깅 용이

### 6.2 실무적 개선 방향
**1. 도메인 특화 최적화**
- 의료: 검색 실패 시 위험도 높음 → 보수적 Ambiguous 임계값
- 금융: 시간 민감도 높음 → 웹 검색 우선순위 상향
- 법률: 정확성 극대화 → 평가기 confidence 강화

**2. 비용-성능 트레이드오프**
- 문제: 웹 검색 API 호출 비용 (Table 6에는 미포함)
- 해결안: 
  - 캐싱 메커니즘: 동일/유사 질문에 대해 검색 결과 재사용
  - 조건부 검색: Incorrect 확률 > 0.3일 때만 웹 검색 수행

**3. 실시간 지식 통합**
- 현재 한계: Wikipedia 기반 정적 정보
- 개선안: 뉴스 API, 날씨 API, 주식 시세 등 실시간 연결
- 효과: 시간 민감도 높은 질문에서 CRAG 성능 향상

### 6.3 평가 벤치마크 개발
**새로운 시나리오 필요**:
1. **의도적 검색 실패**: 모든 상위-10 문서가 부관련 (실제 성능 측정)
2. **도메인 이동 (Domain Shift)**: 의료 도메인 학습 → 법률 도메인 평가
3. **분포 외 데이터**: 학습 시 보지 못한 주제 (OOD, Out-of-Distribution)

**메트릭 확대**:
- 현재: 정확도, FactScore, F1-점수
- 추가 제안: 
  - 신뢰도 캘리브레이션 (Brier Score)
  - 견고성 점수 (RARE 프레임워크)
  - 레이턴시-정확도 곡선 (Pareto frontier)

### 6.4 윤리 및 사회적 고려
**1. 웹 검색의 편향성**
- 위험: 검색 결과가 주류 의견 편향
- 대안: 다양한 관점 소스 통합
- 예시: Correct 행동도 웹 검색으로 보완 고려

**2. 출처 추적의 중요성**
- 현재: Wikipedia 페이지 우선하나, 최종 생성에서 출처 명시 미약
- 개선안: 생성된 각 문장마다 출처 표시 (인용 정확도)

**3. 개인정보 보호**
- 위험: 웹 검색으로 사용자 질문이 외부로 노출
- 환경: 엔터프라이즈 환경에서는 폐쇄 네트워크 필수
- 해결안: 온프레미스 검색 엔진과의 통합 옵션

## 결론: CRAG의 학술적 의의 및 영향
**CRAG**는 2024년 RAG 연구에서 **패러다임 전환을 제시한 논문**으로 평가된다. 기존 RAG 개선 연구들(Self-RAG, RAFT)이 "더 나은 검색" 또는 "더 나은 생성"에 집중했다면, CRAG는 "검색 실패 상황에의 대응"이라는 근본적으로 다른 질문을 제기했기 때문이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b32b600a-39d4-4b53-8167-822567a79bdf/2401.15884v3.pdf)

**핵심 성과**:
- 4개 이질적 데이터셋에서 평균 10% 이상의 안정적 성능 향상
- LLM 변경에 대한 우수한 적응성 (Self-RAG 대비 25포인트 덜 하락)
- 검색 품질 저하에 대한 견고성 입증 (검색 정확도 70% → 20% 시 25포인트 하락 vs Self-RAG 34포인트)

**향후 영향**:
- 웹 검색 통합이 RAG의 표준 아키텍처로 자리잡을 가능성 높음
- 검색 평가(Retrieval Evaluation)가 RAG 최적화의 핵심 축으로 인정될 것으로 예측
- RARE(2025)와 같은 견고성 평가 프레임워크의 표준으로 CRAG 기법이 참조될 가능성

**한계 극복의 방향**:
- 적응형 임계값 학습으로 수동 튜닝 제거
- 평가 기능의 LLM 내재화로 경량화
- 멀티홉 추론으로 복잡한 질의 처리 향상

AI 분야 연구자들은 이 논문의 사상에서 **"실패에 우아하게 대응하는" 시스템 설계의 중요성**을 배울 수 있을 것이다.

***

## 참고 자료

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2401.15884v3.pdf

[^1_2]: https://www.semanticscholar.org/paper/ddbd8fe782ac98e9c64dd98710687a962195dd9b

[^1_3]: https://arxiv.org/html/2401.15269v2

[^1_4]: http://arxiv.org/pdf/2412.02563.pdf

[^1_5]: https://arxiv.org/html/2411.19443v1

[^1_6]: https://arxiv.org/html/2407.19813v3

[^1_7]: http://arxiv.org/pdf/2501.08248.pdf

[^1_8]: http://arxiv.org/pdf/2410.13192.pdf

[^1_9]: https://arxiv.org/pdf/2403.18243.pdf

[^1_10]: http://arxiv.org/pdf/2501.14342.pdf

[^1_11]: https://arxiv.org/abs/2310.11511

[^1_12]: https://openreview.net/pdf?id=rzQGHXNReU

[^1_13]: https://arxiv.org/html/2509.03787v1

[^1_14]: https://openreview.net/attachment?id=hSyW5go0v8\&name=pdf

[^1_15]: https://arxiv.org/abs/2403.10131

[^1_16]: https://arxiv.org/abs/2410.12837

[^1_17]: https://www.youtube.com/watch?v=BUFlyIr3Cxw

[^1_18]: https://ai.meta.com/blog/raft-llama-retrieval-augmented-generation-supervised-fine-tuning-microsoft/

[^1_19]: https://openreview.net/forum?id=ZS4m74kZpH

[^1_20]: https://www.semanticscholar.org/paper/Self-RAG:-Learning-to-Retrieve,-Generate,-and-Asai-Wu/ddbd8fe782ac98e9c64dd98710687a962195dd9b

[^1_21]: https://www.youtube.com/watch?v=cbQ5rm1jOuU

[^1_22]: https://aclanthology.org/2024.emnlp-main.249/

[^1_23]: https://openreview.net/forum?id=hSyW5go0v8

[^1_24]: https://openreview.net/forum?id=rzQGHXNReU

[^1_25]: https://liner.com/review/rare-retrievalaware-robustness-evaluation-for-retrievalaugmented-generation-systems

[^1_26]: https://arxiv.org/pdf/2403.10131.pdf

[^1_27]: https://arxiv.org/html/2506.00054v1

[^1_28]: https://arxiv.org/html/2410.17952v1

[^1_29]: https://www.arxiv.org/pdf/2506.00789.pdf

[^1_30]: https://arxiv.org/html/2506.10408v1

[^1_31]: https://arxiv.org/pdf/2510.11217.pdf

[^1_32]: https://arxiv.org/html/2403.10131v1

[^1_33]: https://arxiv.org/html/2507.18910v1

[^1_34]: https://arxiv.org/pdf/2403.10131v1.pdf

[^1_35]: https://arxiv.org/abs/2405.20978

[^1_36]: https://galileo.ai/blog/raft-adapting-llm

[^1_37]: https://arxiv.org/html/2403.10131v2
