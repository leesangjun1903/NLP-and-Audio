

# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

## 1. 논문 개요 및 핵심 주장

**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**은 2023년 10월 arXiv에 발표된 논문으로, Akari Asai 등이 University of Washington, Allen Institute for AI, IBM Research에서 작성했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

Self-RAG는 대규모 언어 모델(LLM)의 근본적인 한계—사실 오류와 환각(hallucination)—를 해결하는 새로운 패러다임을 제시합니다. 기존의 Retrieval-Augmented Generation(RAG) 방식이 고정된 수의 문서를 무조건 검색하여 모델의 다재다능성을 저해한다는 문제점을 지적합니다. Self-RAG의 핵심 주장은 다음과 같습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

**첫째, 적응형 검색의 필요성**: 모든 작업에 외부 지식이 필요하지는 않습니다. 예를 들어 "당신의 최고의 여름 휴가를 설명하라"는 개인적인 경험에 관한 질문으로, 외부 검색 없이 매개변수 지식만으로 충분합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

**둘째, 생성물의 자기 평가 필요성**: LLM이 생성한 텍스트가 검색된 문서와 일치하고 사실에 부합하는지 명시적으로 평가해야 합니다.

**셋째, 인퍼런스 시점 제어 가능성**: 반영 토큰(reflection tokens)을 활용하여 추론 단계에서 모델의 동작을 커스터마이징할 수 있어야 합니다.

## 2. 해결하고자 하는 문제

### 2.1 기존 RAG의 한계

기존의 표준 RAG 방식(Lewis et al., 2020; Guu et al., 2020)은 세 가지 핵심 문제를 갖고 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

1. **과도한 검색**: 검색 필요 여부를 판단하지 않고 모든 입력에 대해 고정된 수의 문서를 검색합니다. 이는 불필요한 문서를 포함시켜 생성 품질을 저하시킵니다.

2. **일관성 부재**: 모델이 검색된 문서를 따르도록 명시적으로 학습되지 않아, 생성 결과가 검색 결과와 불일치할 수 있습니다.

3. **낮은 투명성**: 생성 과정에 대한 자기 평가 메커니즘이 없어 인용 정확도가 낮습니다.

### 2.2 문제점의 실제 사례

Table 1에 나타난 예시를 보면, 기존 RAG가 "미국 주의 이름은 어떻게 생겼는가?"라는 질문에 대해 아무 관련 없는 "텍사스의 인기 아기 이름: 엠마"라는 문서를 검색하는 등 비효율적입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

## 3. 제안하는 방법

### 3.1 Reflection 토큰의 정의 및 역할

Self-RAG는 네 가지 유형의 reflection 토큰을 도입합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

| 토큰 유형 | 입력 | 출력 | 정의 |
|----------|------|------|------|
| **Retrieve** | x / x, y | {yes, no, continue} | 검색 필요 여부 결정 |
| **IsREL** | x, d | {relevant, irrelevant} | 검색된 문서 관련성 평가 |
| **IsSUP** | x, d, y | {fully, partially, no} | 생성물이 문서로 지지되는 정도 |
| **IsUSE** | x, y | {5,4,3,2,1} | 생성물의 전체 유용성 평가 |

### 3.2 핵심 수식

**Critic 모델 학습**:
$$\max_C E_{(x,y),r \sim D_{critic}} \log p_C(r|x, y)$$

여기서 $r$은 반영 토큰입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

**Generator 모델 학습**:
$$\max_M E_{(x,y,r) \sim D_{gen}} \log p_M(y, r|x)$$

기존 LLM 목표와 동일하나, 모델이 원래 어휘와 함께 새로운 반영 토큰도 예측합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

**세그먼트 수준 빔 서치 점수**:
$$f(y_t, d, \text{Critique}) = p(y_t|x, d, y_{<t}) + S(\text{Critique})$$

여기서:
$$S(\text{Critique}) = \sum_{G \in G} w_G s_G^t$$

$G = \{\text{IsREL}, \text{IsSUP}, \text{IsUSE}\}$는 비판 토큰 그룹이고, $s_G^t$는 가장 바람직한 토큰의 정규화된 확률입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

$$s_G^t = \frac{p_t(\hat{r})}{\sum_{i=1}^{N_G} p_t(r_i)}$$

가중치 $w_G$는 하이퍼파라미터로 인퍼런스 시점에 조정 가능합니다.

### 3.3 모델 구조 및 훈련 절차

Self-RAG는 두 단계로 구성됩니다:

**Step 1: Critic 모델 학습** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

1. GPT-4를 사용하여 4-20k 감독 데이터 생성 (반영 토큰 타입별)
2. Llama2-7B로 초기화한 Critic 모델을 조건부 언어 모델링으로 학습
3. 결과: 90% 이상의 GPT-4 예측과의 일치도

**Step 2: Generator 모델 학습** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

1. 원래의 input-output 쌍에 대해 Critic 모델을 실행
2. 각 세그먼트마다:
   - Retrieve 토큰 예측 (검색 필요 여부)
   - 필요 시 K개 문서 검색
   - 각 문서에 대해 IsREL, IsSUP 예측
   - 전체 문서 후 IsUSE 예측
3. 반영 토큰이 삽입된 코퍼스(150k 샘플)로 표준 다음 토큰 목표로 학습

### 3.4 인퍼런스 알고리즘

Algorithm 1은 Self-RAG의 인퍼런스 절차를 보여줍니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

1. 입력 $x$와 이전 생성 $y_{<t}$가 주어지면, 모델이 Retrieve 토큰 예측
2. Retrieve = Yes인 경우:
   - 검색기 $R$을 사용하여 상위 K개 문서 검색
   - 각 문서 $d$에 대해 IsREL, 다음 응답 세그먼트, IsSUP, IsUSE 예측
   - 세 가지 비판 토큰 확률에 기반한 선형 조합으로 순위 매김
3. Retrieve = No인 경우:
   - 검색 없이 $y_t$ 생성 및 IsUSE 예측

## 4. 성능 향상 분석

### 4.1 전체 평가 결과

Table 2에 따르면, Self-RAG는 6개 다양한 태스크에서 상당한 성능 향상을 보입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

**단순 설정 (Closed-set) 및 단문 생성 (Short-form)**

- **PopQA (단문 질답)**: 55.8% (13B) vs ChatGPT 29.3% → **91% 상대 향상**
- **TriviaQA (단문 질답)**: 69.3% (13B) vs ChatGPT 74.3% → 경합하는 성능
- **PubHealth (사실 검증)**: 74.5% (13B) vs ChatGPT 70.1% → **+4.4%p**
- **ARC-Challenge (다중 선택)**: 73.1% (13B) vs ChatGPT 75.3% → 거의 동등

**장문 생성 및 인용**

- **Biography (FactScore 기준)**: 80.2 (13B) vs ChatGPT 71.8 → **+8.4 포인트**
- **ASQA (인용 정밀도)**: 70.3% (13B) vs ChatGPT 65.1% → **+5.2%p** (ChatGPT는 65.1%, 76.6%)

### 4.2 기존 RAG와의 비교

검색 없는 모델 대비 Self-RAG의 향상도: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

| 데이터셋 | Llama2-7B | Alpaca-7B | Self-RAG 7B | 향상 |
|---------|----------|----------|-----------|------|
| PopQA   | 38.2%    | 46.7%    | 54.9%     | +17% |
| PubHealth | 30.0%  | 40.2%    | 72.4%     | +42% |
| ASQA (정밀도) | 2.9% | 5.5% | 66.9% | +60% |

### 4.3 Ablation 연구

Figure 3a의 Ablation 결과는 각 구성 요소의 중요성을 입증합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- **No Retriever**: 기본 지시 튜닝, 성능 크게 저하
- **No Critic**: 반영 토큰 없이 상단 1개 문서만 사용 → 18.1% vs 30.6% (ASQA)
- **Hard Constraints**: 적응형 임계값 대신 항상 Retrieve=Yes 사용 → 28.3% vs 45.5% (PopQA)

이는 **자기 반성 메커니즘이 성능의 핵심**임을 보여줍니다.

## 5. 모델의 일반화 성능 향상

### 5.1 강점

**데이터 효율성**

Figure 4a-c는 훈련 데이터 크기의 영향을 분석합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- 50k 샘플에서: PopQA 45.5%, Llama2-FT (동일 데이터)는 48.7% vs Self-RAG 45.5%
- 150k 샘플에서: 모든 데이터셋에서 지속적인 개선
- **원외 데이터(Out-of-domain)에서도 견고**: TriviaQA에서 50k 학습으로도 이미 66.4% 달성

**Zero-shot 일반화**

Self-RAG는 학습 과정에서 본적 없는 태스크들에서도 강력한 성능을 보입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- 250개 태스크의 다양한 지시 튜닝 데이터(Open-Instruct) 포함
- 모든 6개 벤치마크 태스크에서 ChatGPT 능가 또는 경합

**인퍼런스 시 커스터마이제이션**

Figure 3b는 반영 토큰 가중치 조정의 효과를 보여줍니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- IsSUP 가중치 증가: Citation precision 71% → 72.5%
- 결과: **추가 훈련 없이** 사용자 요구사항에 맞춤 가능

### 5.2 약점 및 한계

**1. Citation-Fluency 트레이드오프** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- IsSUP 가중치 증가 시: Citation precision ↑ 그러나 MAUVE (유창성) ↓
- Fully supported 답변은 일반적으로 더 짧고 간결함

**2. 모델 크기의 역설적 결과** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- 기이하게도 7B 모델이 13B 모델보다 일부 정밀도 지표에서 우수
- 기술적 원인: 더 작은 모델이 더 신중한 출력 생성

**3. 검색기 성능 의존성** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- Contriever-MS MARCO 검색기 성능에 크게 의존
- Biography, Open-domain QA: Google 검색 API 보강 필요
- ASQA: GTR-XXL (저자 제공) 인덱스 사용

**4. Reflection 토큰 정확도 한계** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

Figure 4d (인간 평가):
- PopQA: IsREL 95%, IsSUP 90% 정확도
- Biography: IsREL 92.5%, IsSUP 70%
- **IsSUP는 인간도 합의하기 어려운 과제** (70% 수준에 머무름)

**5. 단순 질문에 대한 과도 검색** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- 때로 "간단한 퀴즈는 필요 없음"이라는 명확한 신호에도 검색 수행
- 일관성 있는 패턴을 찾기 어려움

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 RAG 진화의 주요 단계

**Phase 1: 기본 RAG (2020-2021)**

- **RAG (Lewis et al., 2020)**: 고정 수의 문서 검색 → 기준선 제시 [arxiv](https://arxiv.org/abs/2312.10997)
- 한계: 무조건 검색, 검색 필요성 판단 없음

**Phase 2: 적응형 검색 (2023-2024)**

- **Active RAG / FLARE (Jiang et al., 2023)**: 동적 검색 결정 [arxiv](https://arxiv.org/abs/2305.06983)
  - 다음 문장 예측으로 쿼리 생성
  - 낮은 신뢰도 토큰 포함 시만 재검색
  - vs Self-RAG: 생성만 하고 자기 평가 없음

- **Augmentation-Adapted Retriever (Yu et al., 2023)**: 범용 플러그인 [aclanthology](https://aclanthology.org/2023.acl-long.136/)
  - 다양한 LM에 적용 가능한 인-컨텍스트 검색
  - Zero-shot 일반화 80%+ 달성
  - vs Self-RAG: 제어 토큰 미활용, 반영 메커니즘 없음

- **Adaptive-RAG (Jeong et al., 2024)**: 쿼리 복잡도 기반 전략 [semanticscholar](https://www.semanticscholar.org/paper/Adaptive-RAG:-Learning-to-Adapt-Retrieval-Augmented-Jeong-Baek/e5e8c6ac537e0f5b5db14170bc232d6f9e641bbc)
  - 쿼리 분류기가 단순/복합 판단
  - 해당 전략 적용 (no-retrieval/single-step/multi-step)
  - vs Self-RAG: 반영 토큰 기반 미세 제어 불가

### 6.2 인용 및 사실성 개선 연구

**인용 정확도 향상**

- **CoVE (Dhuliawala et al., 2023)**: 반복적 검증 체인
- **IRCOT (기타 논문)**: 체인-오브-생각 기반 순환 검색
- Self-RAG 우위: IsSUP 토큰으로 문장 단위 인용 추적 → 사실 검증 자동화

**Domain-specific RAG**

- **의료 분야**: FactScore 개선, Citation precision 70% 이상 (Self-RAG) [pubs.rsna](http://pubs.rsna.org/doi/10.1148/ryai.240313)
- **방사선 영상**: RAG로 81.2% vs 75.5% (기본) 개선 [pubs.rsna](http://pubs.rsna.org/doi/10.1148/ryai.240313)

### 6.3 최신 발전 방향 (2024-2025)

**1. Graph-based RAG**

- **GlobalRAG (2025)**: 구조적 일관성 유지 [arxiv](https://arxiv.org/html/2510.26205v1)
  - 다중홉 추론: 기존 RAG 1.51 F1 → 6.63 F1
  - vs Self-RAG: 선형 검색-생성만 수행

**2. Multimodal 확장**

- **Augmenting Multimodal LLMs (2024)**: 자기 반영 토큰을 이미지/비디오로 확장 [arxiv](https://arxiv.org/html/2411.16863v1)
- 가정: Self-RAG의 반영 메커니즘이 강력한 기초 제공

**3. 효율성 개선 (2025)**

- **Asynchronous Retrieval Pipelines**: 검색-생성을 병렬화 [chitika](https://www.chitika.com/retrieval-augmented-generation-rag-the-definitive-guide-2025/)
- **Hardware-aware Optimization**: GPU/TPU 효율적 활용
- Self-RAG는 이미 병렬 문서 처리이나 추가 최적화 여지

**4. 강화 학습 기반 최적화**

- **MBA-RAG (2024)**: Multi-Armed Bandit으로 적응형 전략 선택 [arxiv](https://arxiv.org/abs/2412.01572)
- **OpenRAG (2025)**: 인-컨텍스트 관련성 학습 [arxiv](https://arxiv.org/pdf/2503.08398.pdf)
- Self-RAG vs RL: Self-RAG는 오프라인 Critic으로 비용 절감, RL은 더 세밀한 최적화

### 6.4 Generalization 관련 핵심 연구

| 논문 | 연도 | Generalization 강점 | Self-RAG와의 차이 |
|------|------|------------------|-----------------|
| **AAR** | 2023 | Zero-shot 80%+, 범용 플러그인 | 도메인별 특화 안 됨 |
| **Adaptive HyDE** | 2025 | Novel question 100% 커버 | 동적 임계값 조정 |
| **Question Decomposition RAG** | 2025 | 다중홉 질문 분해 | 선형 검색만 가능 |
| **LDRA** | 2025 | 다양성 보존하며 검색 | 다양성 고려 없음 |
| **Self-RAG** | 2023 | 반영 토큰 커스터마이제이션 | 실시간 가중치 조정 |

**Generalization 우수 사례:**

- **범용성**: Self-RAG는 Llama2, Alpaca 기반이지만 모든 벤치마크에서 작동
- **계산 효율**: Adaptive retrieval로 PopQA 검색 빈도 조정 가능 (0.2→0.6 임계값)
- **다중 도메인**: 지시 튜닝 데이터의 다양성(11개 데이터셋) → 견고한 일반화

## 7. 논문이 앞으로의 연구에 미치는 영향

### 7.1 학문적 영향

**1. 패러다임 전환**

Self-RAG는 RAG 분야에서 세 가지 인식 전환을 주도했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)

- 검색 → **선택적 검색 + 자기 평가**로의 전환
- 고정 파이프라인 → **인퍼런스 시 제어 가능**한 구조
- 모델 평가 → **반영 토큰 기반 자동 평가** 가능성

**2. 후속 연구 활성화**

2023년 10월 이후 주요 논문들이 Self-RAG의 아이디어를 확장:

- Multimodal LLM에 self-reflective 토큰 적용 [arxiv](https://arxiv.org/html/2411.16863v1)
- Graph 구조와 결합한 GlobalRAG [arxiv](https://arxiv.org/html/2510.26205v1)
- 강화 학습과의 통합 연구 진행중

### 7.2 산업 적용 시사점

**1. 비용 절감** [morphik](https://www.morphik.ai/blog/retrieval-augmented-generation-strategies)

- RAG 시장 2024년 $1.85B, CAGR 49%
- Self-RAG의 적응형 검색: 불필요한 API 호출 감소
- 예: PopQA에서 검색 빈도 25% → 실행 시간 30% 단축 가능

**2. 투명성 요구 충족**

의료, 법률 등 규제 산업에서:

- **의료**: Citation precision 70.3% (Self-RAG) → 의사 신뢰도 향상 [pubs.rsna](http://pubs.rsna.org/doi/10.1148/ryai.240313)
- **법률**: 인용 출처 명확화로 사용자 검증 가능

**3. 도메인 특화 시스템**

- 의료 RAG: FactScore 개선, 오류율 감소 [pubs.rsna](http://pubs.rsna.org/doi/10.1148/ryai.240313)
- 라디오로지: Self-RAG 기반 임상 응용 개발 진행 [pubs.rsna](http://pubs.rsna.org/doi/10.1148/ryai.240313)

### 7.3 한계 및 향후 연구 방향

**1. 일반화 성능 한계 극복**

- **문제**: 특정 도메인(예: 의료)에서 추가 학습 필요
- **해결책**: Few-shot adaptation, Domain-specific Critic 모델 개발
- **관련 연구**: ALoFTRAG (자동 로컬 파인튜닝, 2025) [arxiv](https://arxiv.org/pdf/2501.11929.pdf)

**2. Hallucination 완전 해결**

- **문제**: Reflection 토큰도 LLM이 생성 → 오류 가능
- **현재 결과**: IsSUP 정확도 70% (인간도 어려워함) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ff036b19-bdaf-40c2-abdb-e467ed93937a/2310.11511v1.pdf)
- **개선 방향**: 
  - 외부 검증기 통합 (예: 온톨로지 검증)
  - Knowledge graph 기반 팩트 체크
  - 다중 검색기 앙상블 활용

**3. Multi-hop 추론 강화**

- **문제**: Self-RAG는 선형 검색-생성만 가능
- **해결책 진행중**:
  - Question decomposition (2025) [arxiv](https://arxiv.org/abs/2507.00355)
  - Graph-based retrieval (GlobalRAG, 2025) [arxiv](https://arxiv.org/html/2510.26205v1)
  - Iterative retrieval-generation synergy [arxiv](https://arxiv.org/abs/2305.15294)

**4. 계산 효율성**

- **병목**: K개 문서에 대한 병렬 생성 오버헤드
- **개선 방향**:
  - Asynchronous pipeline (2025) [chitika](https://www.chitika.com/retrieval-augmented-generation-rag-the-definitive-guide-2025/)
  - Speculative decoding
  - 지연 시간 50-200ms 목표

**5. 다국어 및 멀티모달 확장**

- **현재**: 영문 데이터셋 중심
- **필요**: 다국어 Critic 모델 개발 (추가 비용)
- **진행**: Multimodal LLM 확장 연구 [arxiv](https://arxiv.org/html/2411.16863v1)

## 결론

Self-RAG는 Retrieval-Augmented Generation 분야에서 **적응형 검색과 자기 반성이라는 이중 메커니즘**을 도입함으로써 기존의 한계를 체계적으로 극복합니다. 7B/13B의 비교적 작은 모델로 ChatGPT를 능가하는 성능을 달성한 것은, 구조적 혁신의 중요성을 입증합니다.

특히 **인퍼런스 시점 제어 가능성**은 사용자의 다양한 요구사항(정확도 vs. 유창성, 속도 vs. 품질)에 대응할 수 있게 하는 획기적인 특징입니다. 다만 검색기 성능 의존성, Reflection 토큰 정확도의 근본적 한계, 단순 다중홉 추론 불가능성 등은 향후 연구의 과제입니다.

2024-2025년의 후속 연구는 Graph 기반 추론, Multimodal 확장, 강화 학습 통합, 그리고 효율성 개선으로 진화하고 있으며, Self-RAG는 이러한 고도화의 **튼튼한 기초**로 자리매김했습니다.

***

## 참고문헌


Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," arXiv:2310.11511, Oct 2023[^1_1]

Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," Syst. \& Health, 2025 (Meta-analysis)[^1_20]

Oxford Academic, "Improving large language model applications in biomedicine with retrieval-augmented generation," JAMIA, Jan 2025[^1_21]

PLOS Digital Health, "Retrieval augmented generation for large language models in medicine," Jun 2025[^1_10]

IEEE Xplore, "Structured Retrieval-Augmented Generation for Multi-Entity Question Answering," May 2025[^1_22]

arXiv:2507.00355, "Question Decomposition for Retrieval-Augmented Generation," Jun 2025[^1_23]

Question Decomposition RAG, 2025 (referenced in )[^1_18]

Springer, "Evaluation of a Retrieval-Augmented Generation-Powered Chatbot," Mar 2025[^1_24]

arXiv:2508.06401, "A Systematic Literature Review of Retrieval-Augmented Generation," Aug 2025[^1_25]

IEEE, "Evaluating Retrieval-Augmented Generation Strategies for Large Language Models in Travel Mode Choice Prediction," Aug 2025[^1_26]

IEEE, "Personalized Music Recommendations Using Retrieval Augmented Generation," Mar 2025[^1_27]

arXiv:2503.08398, "OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning," Mar 2025[^1_15]

arXiv:2405.13576, "FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research," 2025[^1_28]

arXiv:2410.12837, "A Comprehensive Survey of Retrieval-Augmented Generation: Evolution, Current Landscape and Future Directions," Oct 2024[^1_29]

arXiv:2406.16828, "Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track," Jun 2024[^1_30]

arXiv:2409.15337, "Revisiting the Solution of Meta KDD Cup 2024: CRAG," Sep 2024[^1_31]

arXiv:2503.02922, "Optimizing open-domain question answering with graph-based retrieval augmented generation," Mar 2025[^1_32]

arXiv:2408.11381, "RAGLAB: A Modular and Research-Oriented Unified Framework for Retrieval-Augmented Generation," Sep 2024[^1_33]

arXiv:2406.12534, "Unified Active Retrieval for Retrieval Augmented Generation," Oct 2024[^1_34]

Morphik.ai, "RAG in 2025: 7 Proven Strategies to Deploy Retrieval-Augmented Generation," Jul 2025[^1_16]

arXiv:2310.11511, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," Oct 2023[^1_35]

Liner, "Augmentation-Adapted Retriever Improves Generalization of Language Models as Generic Plug-In," 2023[^1_36]

Chitika, "Retrieval-Augmented Generation (RAG): The Definitive Guide ," Jan 2025[^1_13]

Tistory, "[논문리뷰] SELF-RAG: LEARNING TO RETRIEVE...," Sep 2024[^1_37]

PyTorch Korea, "대규모 언어 모델을 위한 검색-증강 생성(RAG) 기술 현황," Jan 2024[^1_38]

EdenAI, "The 2025 Guide to Retrieval-Augmented Generation (RAG)," Jan 2025[^1_39]

Velog, "Self-RAG: LEARNING TO RETRIEVE, GENERATE, AND...", 2023[^1_40]

Velog, "[논문 리뷰] Adaptive-RAG: Learning to Adapt Retrieval...", 2024[^1_41]

Data Nucleus, "RAG in 2025: The enterprise guide to retrieval augmented," Jan 2026[^1_42]

Tistory, "[LLM] SELF-RAG: Learning to Retrieve, Generate and Critique...", 2023[^1_43]

ACL Anthology, "Augmentation-Adapted Retriever Improves Generalization," 2023[^1_5]

arXiv:2312.10997, "Retrieval-Augmented Generation for Large Language Models: A Survey," Dec 2023[^1_2]

arXiv:2510.26205, "Towards Global Retrieval Augmented Generation," Sep 2025[^1_11]

arXiv:2505.24726, "Self-Improving LLMs via Reinforcement Learning," May 2025[^1_44]

arXiv:2506.00054, "Retrieval-Augmented Generation: A Comprehensive Survey," May 2025[^1_45]

arXiv:2411.16863, "Augmenting Multimodal LLMs with Self-Reflective Tokens," Nov 2024[^1_12]

arXiv:2305.17331, "Augmentation-Adapted Retriever Improves Generalization," 2023[^1_6]

PLOS, "Retrieval augmented generation for large language models in health," Jun 2025[^1_46]

arXiv:2411.16863, "Wait, We Don't Need to 'Wait'! Removing Thinking Tokens," 2024[^1_47]

arXiv:2501.12835, "Adaptive Retrieval without Self-Knowledge," 2025[^1_48]

arXiv:2410.12837, "A Comprehensive Survey of Retrieval-Augmented Generation," Oct 2024[^1_49]

arXiv:2406.10400, "Self-Reflection Makes Large Language Models Safer," 2024[^1_50]

arXiv:2403.14403, "Learning to Adapt Retrieval-Augmented Large Language Models," Mar 2024[^1_51]

Tistory, "[논문 리뷰] SELF-RAG: LEARNING TO RETRIEVE...," Aug 2025[^1_52]

Tistory, "[논문리뷰] Adaptive-RAG: Learning to Adapt Retrieval," 2024[^1_53]

arXiv:2305.06983, "Active Retrieval Augmented Generation," May 2023[^1_3]

arXiv:2311.04177, "Enhancing LLM Intelligence with ARM-RAG," Nov 2023[^1_54]

Semantic Scholar, "ReZG: Retrieval-Augmented Zero-Shot Counter Narrative Generation," Oct 2023[^1_55]

arXiv:2305.15294, "Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy," May 2023[^1_19]

arXiv:2310.19998, "Generative retrieval-augmented ontologic graph and multi-agent strategies," Oct 2023[^1_56]

ACL Anthology, "Team NLLG submission for Eval4NLP 2023," 2023[^1_57]

arXiv:2311.18397, "IAG: Induction-Augmented Generation Framework," Nov 2023[^1_58]

Semantic Scholar, "Prompt Generate Train (PGT): A framework," 2023[^1_59]

arXiv:2308.04711, "Answering Unseen Questions With Smaller Language Models," Aug 2023[^1_60]

IEEE, "UniPoll: A Unified Social Media Poll Generation Framework," Jun 2023[^1_61]

ACL Anthology PDF, "Active Retrieval Augmented Generation," May 2023[^1_62]

ACL Anthology, "Retrieving Multimodal Information for Augmented Generation," Mar 2023[^1_63]

ACM, "AU-RAG: Agent-based Universal Retrieval Augmented Generation," Dec 2024[^1_64]

arXiv PDF, "Active Retrieval Augmented Generation," Oct 2023[^1_65]

ACL Anthology PDF, "Enhancing Retrieval-Augmented Large Language Models," May 2023[^1_66]

arXiv, "AT-RAG: An Adaptive RAG Model Enhancing Query Efficiency," Oct 2024[^1_67]

arXiv:2501.11929, "ALoFTRAG: Automatic Local Fine Tuning for RAG," Jan 2025[^1_17]

Semantic Scholar, "Adaptive-RAG: Learning to Adapt Retrieval-Augmented," 2024[^1_7]

arXiv:2506.00054, "Retrieval-Augmented Generation - A Comprehensive Survey," May 2025[^1_68]

ACL Anthology PDF, "Learning to Adapt Retrieval-Augmented Large Language Models," Jun 2024[^1_8]

ACL Anthology PDF, "Active Retrieval Augmented Generation," Dec 2023[^1_69]

ACL Anthology, "Learning to Adapt Retrieval-Augmented Large Language Models," Jun 2024[^1_9]

SSRN, "Retrieval-Augmented Generation: A Comprehensive Survey," Dec 2024[^1_70]

ACL Anthology, "Active Retrieval Augmented Generation," Dec 2023[^1_4]

arXiv, "Adaptive-RAG: Learning to Adapt Retrieval-Augmented," Mar 2024[^1_71]

Blog OUTTA.ai, "[2024-2] 백승우 - Retrieval-Augmented Generation," Nov 2024[^1_72]

Liner, "[Quick Review] Active Retrieval Augmented Generation," Aug 2025[^1_73]

GitHub, "starsuzi/Adaptive-RAG," Apr 2024[^1_74]

Velog, "[논문 리뷰] Retrieval-Augmented Generation for Large Language Models," 2024[^1_75]

arXiv, "A Survey on Recent Advances in LLM-Based Multi-turn," Jul 2023[^1_76]

arXiv PDF, "arXiv:2403.14403v2 - Adaptive-RAG," Mar 2024[^1_77]

arXiv PDF, "Retrieval Diversity Boosts Multi-Turn Intent Understanding," Oct 2025[^1_78]

arXiv, "A Memory-Active Policy for Multi-Session Task-Oriented," Oct 2010[^1_79]

arXiv PDF, "arXiv:2311.05085v2," Jan 2024[^1_80]

arXiv, "MBA-RAG: a Bandit Approach for Adaptive Retrieval," 2024[^1_14]

arXiv PDF, "A Representation Sharpening Framework," Nov 2025[^1_81]

arXiv, "Learning to Adapt Retrieval-Augmented Large Language Models," Mar 2024[^1_82]

arXiv, "Evaluation of Retrieval-Augmented Generation: A Survey," 2024[^1_83]
