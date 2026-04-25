
# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

> **논문 정보**: Patrick Lewis et al., NeurIPS 2020
> **arXiv**: [2005.11401](https://arxiv.org/abs/2005.11401)
> **발표**: 34th Conference on Neural Information Processing Systems (NeurIPS 2020)

---

## 1. 📌 핵심 주장 및 주요 기여 요약

대형 사전 학습 언어 모델(LLM)은 파라미터 내에 사실적 지식을 저장하고 다운스트림 NLP 태스크에서 SOTA를 달성하지만, 지식 집약적(knowledge-intensive) 태스크에서는 지식 접근 및 정밀한 조작 능력이 제한적이고, 의사결정의 출처 제공이나 세계 지식 업데이트가 미해결 문제로 남아 있습니다.

이를 해결하기 위해 Lewis et al.은 다음의 핵심 프레임워크를 제안합니다:

저자들은 사전 학습된 파라메트릭(parametric) 메모리와 비파라메트릭(non-parametric) 메모리를 결합하여 언어 생성을 수행하는 RAG(Retrieval-Augmented Generation)에 대한 범용 파인튜닝 레시피를 탐색합니다.

### 주요 기여 요약표

| 기여 항목 | 내용 |
|---|---|
| **아키텍처 통합** | DPR(검색기) + BART(생성기)의 엔드-투-엔드 결합 |
| **두 가지 변형 모델** | RAG-Sequence, RAG-Token |
| **범용 파인튜닝** | 다양한 NLP 태스크에 적용 가능한 통합 학습법 |
| **비파라메트릭 메모리** | 재훈련 없이 인덱스 교체로 지식 업데이트 가능 |
| **SOTA 달성** | 3개의 Open-domain QA 태스크에서 최고 성능 |

---

## 2. 🔍 상세 분석

### 2-1. 해결하고자 하는 문제

RAG는 전통적인 AI 모델의 핵심 한계, 즉 훈련 데이터 외부에 저장된 최신 또는 특정 정보에 접근할 수 없는 문제를 해결하기 위해 도입되었습니다. 이 접근법은 사전 학습 모델 지식(파라메트릭)과 위키피디아 같은 외부 데이터 소스(비파라메트릭)라는 두 가지 메모리 유형을 통합합니다.

명시적 비파라메트릭 메모리에 대한 미분 가능한 접근 메커니즘을 가진 사전 학습 모델이 이 문제를 극복할 수 있지만, 지금까지는 추출형(extractive) 다운스트림 태스크에만 연구되어 왔습니다.

---

### 2-2. 제안 방법 및 핵심 수식

#### 🔷 모델 구성 요소

RAG 모델의 파라메트릭 메모리는 사전 학습된 seq2seq 트랜스포머이고, 비파라메트릭 메모리는 사전 학습된 신경 검색기로 접근하는 위키피디아의 밀집 벡터 인덱스입니다. 이 구성 요소들은 확률적 모델로 엔드-투-엔드 학습되며, 검색기(DPR)는 입력에 조건화된 잠재 문서를 제공하고 seq2seq 모델(BART)은 이 잠재 문서와 입력을 함께 조건화하여 출력을 생성합니다.

#### 🔷 RAG-Sequence 수식

RAG-Sequence는 전체 출력 시퀀스에 대해 동일한 문서를 사용합니다. Top- $K$ 문서에 대해 주변화(marginalization)하는 수식은 다음과 같습니다:

$$p_{\text{RAG-Sequence}}(y \mid x) \approx \sum_{z \in \text{top-}K} p_\eta(z \mid x) \cdot p_\theta(y \mid x, z)$$

여기서:
- $x$: 입력 쿼리
- $z$: 검색된 문서 (잠재 변수)
- $p_\eta(z \mid x)$: 검색기(DPR)의 문서 확률 (파라미터 $\eta$)
- $p_\theta(y \mid x, z)$: 생성기(BART)의 조건부 확률 (파라미터 $\theta$)

#### 🔷 RAG-Token 수식

RAG-Token은 각 목표 토큰을 예측하기 위해 서로 다른 문서를 사용할 수 있습니다. 각 토큰 $y_i$에 대해 문서를 주변화하는 수식은:

$$p_{\text{RAG-Token}}(y \mid x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}K} p_\eta(z \mid x) \cdot p_\theta(y_i \mid x, z, y_{1:i-1})$$

#### 🔷 검색기(DPR) 수식

검색기의 문서 확률은 쿼리 벡터 $q(x)$와 문서 벡터 $d(z)$의 내적(inner product)으로 계산됩니다:

$$p_\eta(z \mid x) \propto \exp\left(d(z)^{\top} q(x)\right)$$

검색 컴포넌트 $p_\eta(z|x)$는 DPR 기반입니다. DPR은 바이-인코더(bi-encoder) 아키텍처를 따르며, top- $k(p_\eta(\cdot|x))$를 구하는 것은 Maximum Inner Product Search(MIPS) 문제로, 서브-선형 시간에 근사적으로 해결될 수 있습니다.

#### 🔷 학습 목적 함수 (음의 로그우도 최소화)

$$\mathcal{L}(\theta, \eta) = -\sum_{j} \log p(y_j \mid x_j)$$

훈련 시 생성기 컴포넌트 $p_\theta(y_i|x, z, y_{1:i-1})$는 어떠한 인코더-디코더 모델로도 모델링될 수 있으며, 논문에서는 400M 파라미터를 가진 사전 학습된 seq2seq 트랜스포머인 BART-large를 사용합니다.

---

### 2-3. 모델 구조

```
┌─────────────────────────────────────────────────────────┐
│                     RAG 모델 구조                        │
├─────────────────────┬───────────────────────────────────┤
│   비파라메트릭 모듈   │        파라메트릭 모듈              │
│  (Non-Parametric)   │       (Parametric)                │
│                     │                                   │
│  ┌───────────────┐  │  ┌────────────────────────────┐  │
│  │  Query Encoder│  │  │        BART-large          │  │
│  │   (DPR-BERT)  │  │  │    (seq2seq Generator)     │  │
│  └──────┬────────┘  │  │    400M parameters         │  │
│         │ q(x)      │  └────────────┬───────────────┘  │
│  ┌──────▼────────┐  │               │                   │
│  │ Document Index│  │               │                   │
│  │ (Wikipedia    │  │               │                   │
│  │  Dense Vec.)  │  │               │                   │
│  └──────┬────────┘  │               │                   │
│         │ MIPS      │               │                   │
│  Top-K Documents z  │               │                   │
└─────────┼───────────┴───────────────┼───────────────────┘
          │         합성 (Concat)      │
          └──────────────┬────────────┘
                         │
                    Output y
```

사전 학습된 검색기(쿼리 인코더 + 문서 인덱스)와 사전 학습된 seq2seq 모델(생성기)을 결합하여 엔드-투-엔드로 파인튜닝합니다. 쿼리 $x$에 대해 MIPS를 사용하여 Top- $K$ 문서 $z_i$를 찾고, 최종 예측 $y$를 위해 $z$를 잠재 변수로 처리하여 서로 다른 문서에 대한 seq2seq 예측을 주변화합니다.

---

### 2-4. 성능 향상

모델을 다양한 지식 집약적 NLP 태스크에서 파인튜닝 및 평가한 결과, 세 가지 Open-domain QA 태스크에서 SOTA를 달성하였으며, 파라메트릭 seq2seq 모델과 태스크별 retrieve-and-extract 아키텍처를 모두 능가했습니다. 또한 언어 생성 태스크에서 RAG 모델이 SOTA 파라메트릭 전용 seq2seq 기준 모델보다 더 구체적이고, 다양하며, 사실적인 언어를 생성한다는 것을 발견했습니다.

2020년 말까지 수 억 개의 파라미터를 가진 RAG 시스템이 110억 개 파라미터를 가진 폐쇄형(closed-book) LM을 능가하여, 하이브리드 파라메트릭 + 비파라메트릭 메모리의 효율성을 입증했습니다.

#### 주요 성능 지표 요약

| 태스크 | 비교 대상 | 결과 |
|---|---|---|
| Natural Questions (NQ) | T5-11B (closed-book) | RAG 초과 달성 |
| TriviaQA | 기존 retrieve-and-extract | SOTA 달성 |
| FEVER (사실 검증) | 복잡 파이프라인 시스템 | 4.3% 이내 차이 |
| MS-MARCO | BART (파라메트릭만) | 더 다양하고 사실적 |

FEVER 분석에서 상위 검색 문서가 71%의 경우에서 gold article이었고, 상위 10개 검색 문서에서는 90%의 경우 gold article이 포함되어 있었습니다.

---

### 2-5. 한계점

1. **도메인 특화 문제**: RAG는 위키피디아 기반 외부 지식 베이스로만 훈련 및 탐색되었으며, 의료나 뉴스 같은 다른 전문 도메인에 최적화되어 있지 않습니다.

2. **지식 베이스 인코딩 고정**: RAG 모델이 다운스트림 QA 태스크를 위해 파인튜닝될 때, 원래 구현은 패시지 인코딩과 외부 지식 베이스를 고정합니다. 이는 외부 지식 베이스를 재인코딩하는 것이 계산적으로 비용이 많이 들고 정교한 구현이 필요하기 때문입니다.

3. **검색 품질의 종속성**: RAG 기술이 환각(hallucination) 완화에 큰 가능성을 보여주지만, RAG 패러다임 자체도 한계가 있으며 구성 요소의 불충분한 능력이 환각 생성에 기여합니다.

4. **멀티홉 추론 한계**: 검색 지연(retrieval latency), 멀티홉(multi-hop) 추론, 충실도(faithfulness) 평가의 엄밀성 등의 도전 과제들이 후속 연구의 과제로 남아 있습니다.

5. **구조적 관계 누락**: RAG는 실제 시나리오에서 한계에 직면하는데, 텍스트 콘텐츠는 고립되어 있지 않고 서로 연결되어 있습니다. 전통적인 RAG는 의미적 유사성만으로 표현될 수 없는 중요한 구조적 관계 지식을 포착하지 못합니다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

### 3-1. 비파라메트릭 메모리 교체를 통한 지식 업데이트

이 논문은 RAG의 비파라메트릭 메모리를 단순히 교체함으로써 RAG의 세계 지식을 업데이트할 수 있음을 보여줍니다. 이는 재훈련 없이도 모델의 지식 기반을 갱신할 수 있다는 점에서 일반화 성능의 핵심 요소입니다.

수식으로 표현하면, 기존 인덱스 $\mathcal{D}\_{old}$를 새로운 인덱스 $\mathcal{D}_{new}$로 교체할 때:

$$p_{\text{updated}}(y \mid x) \approx \sum_{z \in \text{top-}K(\mathcal{D}_{new})} p_\eta(z \mid x) \cdot p_\theta(y \mid x, z)$$

이를 통해 **Zero-shot 도메인 전이(domain transfer)** 가 가능해집니다.

### 3-2. 다양한 태스크에서의 범용성

KILT 벤치마크는 공유된 위키피디아 스냅샷으로 11개의 지식 집약적 태스크를 통합했습니다. 단일 RAG 기준선이 QA, 대화, 사실 검증, 슬롯 채우기(slot-filling)에서 경쟁력 있는 성능을 보여, RAG의 도메인 불가지론적(domain-agnostic) 가능성을 강조했습니다.

### 3-3. 일반화 성능 향상을 위한 수식적 분석

RAG의 일반화 능력은 다음 분해(decomposition)로 이해할 수 있습니다:

$$p(y \mid x, \mathcal{D}) = \sum_{z \in \mathcal{D}} p_\eta(z \mid x) \cdot p_\theta(y \mid x, z)$$

- **$p_\eta(z \mid x)$**: 검색기의 일반화 → 새로운 도메인의 문서를 효과적으로 검색하는 능력
- **$p_\theta(y \mid x, z)$**: 생성기의 일반화 → 검색된 문서를 활용하여 유연한 답변 생성

T5나 BART처럼, RAG는 어떤 seq2seq 태스크에서도 파인튜닝될 수 있으며, 이때 생성기와 검색기가 함께 학습됩니다.

### 3-4. 도메인 적응(Domain Adaptation) 연구

RAG-end2end 연구의 주요 발견은 검색기 컴포넌트의 적응이 RAG 계열 아키텍처의 전반적인 도메인 적응 성능에 결정적인 역할을 한다는 것입니다. 지식 베이스 인코딩을 업데이트하지 않고 질문 인코더만 파인튜닝하면 성능이 저하될 수 있으며, DPR 검색기를 독립적으로 파인튜닝하는 대신 RAG-end2end 메커니즘의 일부로 파인튜닝하는 것이 더 나은 전반적인 결과를 제공합니다.

---

## 4. 📊 2020년 이후 관련 최신 연구 비교 분석

### 4-1. 주요 후속 연구 비교표

| 연구 | 연도 | 핵심 혁신 | RAG 대비 개선점 |
|---|---|---|---|
| **REALM** (Guu et al.) | 2020 | 마스크 언어 모델링으로 검색기를 사전 학습 단계부터 엔드-투-엔드 학습 | 사전 학습 단계에서 검색 통합 |
| **FiD** (Izacard & Grave) | 2021 | T5 디코더가 수십 개의 검색 패시지를 동시에 처리 | 다수 문서 융합으로 QA 정확도 향상 |
| **ATLAS** (Izacard et al.) | 2022 | 대규모 파라미터 효율적 하이브리드 | 더 작은 모델로 더 큰 모델 능가 |
| **RETRO** (Borgeaud et al.) | 2022 | 조 단위 토큰 코퍼스에서 청크 단위 검색 | 초대규모 비파라메트릭 메모리 |
| **Self-RAG** (Asai et al.) | 2023 | 자기 반성(self-reflection)으로 검색 필요성 판단 | 적응적 검색, 더 정확한 출력 |
| **GraphRAG** | 2023~2024 | 지식 그래프 기반 구조적 검색 | 관계형 지식 포착 |
| **HippoRAG 2** | 2025 | PersonalizedPageRank + 지식 그래프 | 연상 기억 7% 향상 |

### 4-2. 각 연구 상세 분석

#### REALM (2020)
REALM은 마스크 토큰을 검색된 증거로 예측하기 위해 미분 가능한 검색기를 언어 모델에 통합한 검색 증강 사전 학습을 도입했습니다. REALM은 기존 LM 대비 QA에서 상당한 향상을 달성하여, 외부 지식 주입이 사전 학습과 파인튜닝 모두에서 도움이 됨을 검증했습니다.

#### Fusion-in-Decoder (FiD, 2021)
FiD는 RAG처럼 $k$개의 패시지를 검색하지만, 이를 연결하여 T5 기반 seq2seq에 모두 공급함으로써 디코더가 여러 문서에 동시에 주목할 수 있게 합니다. 이 아키텍처는 대형 생성 모델이 많은 패시지에서 증거를 효과적으로 종합할 수 있음을 보여주며, 오픈 QA 벤치마크에서 추가적인 성능 향상을 달성했습니다.

#### RETRO & ATLAS (2022)
RETRO와 ATLAS 같은 파라미터 효율적 하이브리드 모델들은 더 큰 모델만으로 더 좋은 지식을 얻는다는 개념에 도전하며, 대신 고품질 검색과 다중 문서 추론이 핵심 레버로 부상합니다. 조 단위 토큰 코퍼스에서의 더 빠른 검색, 미분 가능한 멀티홉 추론, 증거 충실도의 강건한 평가 등이 여전히 열린 도전 과제입니다.

#### Self-RAG (2023)
Self-RAG는 질문 답변 및 사실 검증에서 검색된 패시지에 대한 과도한 또는 과소한 의존 문제를 해결하며, 생성된 출력에서 환각 같은 문제를 야기할 수 있는 문제를 다룹니다.

#### GraphRAG (2023~2024)
Graph Retrieval-Augmented Generation(GraphRAG)은 이러한 도전 과제들을 해결하는 혁신적인 솔루션으로 등장했습니다. 전통적인 RAG와 달리, GraphRAG는 사전 구축된 그래프 데이터베이스에서 주어진 쿼리에 관련된 관계형 지식을 포함한 그래프 요소를 검색합니다. 이 요소들에는 노드, 트리플, 경로 또는 서브그래프가 포함될 수 있으며, GraphRAG는 텍스트 간의 상호 연결을 고려하여 관계형 정보를 더 정확하고 포괄적으로 검색할 수 있게 합니다.

#### HippoRAG 2 (2025)
지속적으로 지식을 획득, 조직화, 활용하는 능력은 AI 시스템이 갖추어야 할 인간 지능의 핵심 특성입니다. LLM의 지속적 학습에서의 도전 과제를 감안할 때, RAG는 새로운 정보를 도입하는 지배적인 방법이 되었지만, 벡터 검색에 대한 의존성이 인간 장기 기억의 동적이고 상호 연결된 특성을 모방하는 능력을 방해합니다.

---

## 5. 🚀 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

2020년의 이러한 발전들은 Lewis et al.의 RAG 공식화로 수렴되었으며, 검색기-리더 아키텍처와 seq2seq 생성을 엔드-투-엔드 프레임워크로 통합했습니다. RAG 모델은 오픈-도메인 QA의 인사이트를 활용하여 위키피디아에서 텍스트 청크를 가져오는 밀집 패시지 검색기와 강력한 seq2seq 생성기(BART)를 사용하여 두 구성 요소를 공동으로 훈련시켰습니다.

Lewis et al.의 2020년 seminal 연구 이후, RAG는 연구 관심과 학술 출판물의 급격한 증가로 표시되는 빠른 발전을 목격했습니다.

논문의 가장 주목할 만한 기여 중 하나는 AI가 생성한 응답을 검색된 사실 데이터에 기반하게 하는 방법입니다. 이 혁신은 지식 집약적 태스크에서 AI 환각의 지속적인 문제를 해결하여 AI 응용 프로그램을 더 신뢰할 수 있게 만들었습니다.

2020년 RAG의 도입은 핵심 마일스톤으로 간주되는데, 이는 검색 증강 아키텍처를 QA를 넘어 외부 지식이 필요한 모든 생성 태스크로 일반화했기 때문입니다.

### 5-2. 앞으로 연구 시 고려할 점

#### 🔶 기술적 고려 사항

1. **멀티홉 추론 강화**
   - RAG는 검색된 텍스트를 입력에 추가하는 컨텍스트 확장을 통해 작동하지만, 외부 지식이 어텐션 메커니즘 내에서 토큰으로 경쟁합니다. 그 결과, 영향이 간접적이고 특히 장문 컨텍스트 및 멀티홉 추론 시나리오에서 불안정합니다.

2. **검색 품질 최적화**
   - RAG 시스템에서 검색된 콘텐츠의 품질이 생성기에 공급되는 정보를 결정합니다. 낮은 콘텐츠 품질은 모델 환각이나 다른 품질 저하의 위험을 높입니다.

3. **동적 검색 트리거링**
   - 점점 더 많은 시스템들이 생성 불확실성, 태스크 복잡성 또는 중간 출력에 조건화하여 언제 어떻게 검색할지를 동적으로 제어합니다. DRAGIN은 엔트로피 기반 신뢰도 신호를 사용하여 토큰 수준에서 검색을 트리거하고, FLARE는 문장 생성 중 낮은 신뢰도 예측을 기반으로 선택적으로 검색합니다.

4. **하이브리드 접근법**
   - 반사실적(counterfactual) 및 적대적 정보를 처리하는 강건성이 RAG에서 측정 및 개선하는 데 중요합니다. RAG와 파인튜닝 모델의 사용을 최적화하는 방법에 대한 연구가 진행 중입니다.

5. **스케일링 법칙 이해**
   - LLM 스케일링 법칙과 이것이 RAG 시스템에 어떻게 적용되는지에 대한 조사는 아직 충분히 이해되지 않았습니다.

#### 🔶 미래 연구 방향

```
RAG 미래 연구 로드맵
├── 1. 검색 메커니즘 고도화
│   ├── 희소(sparse) + 밀집(dense) 하이브리드 검색
│   ├── 지식 그래프 통합 (GraphRAG)
│   └── 멀티모달 검색 (텍스트 + 이미지)
│
├── 2. 생성 품질 향상
│   ├── 자기 반성 메커니즘 (Self-RAG)
│   ├── Chain-of-Thought + RAG 결합
│   └── 환각 감지 및 수정 모듈
│
├── 3. 도메인 적응
│   ├── 도메인 특화 파인튜닝 (RAFT)
│   ├── 지속적 학습 (Continual Learning)
│   └── 비파라메트릭 연속 업데이트
│
├── 4. 효율성 최적화
│   ├── 경량화 검색 메커니즘
│   ├── 분산 인덱스 시스템
│   └── 추론 지연(latency) 최소화
│
└── 5. 신뢰성 및 보안
    ├── 프라이버시 보존 RAG
    ├── 적대적 공격 방어
    └── 출처 추적(provenance tracking)
```

향후 연구에서 두 컴포넌트를 처음부터 공동으로 사전 학습할 수 있는지 탐색하는 것이 유익할 수 있으며, BART와 유사한 디노이징 목표 또는 다른 목표를 사용할 수 있습니다. 이 연구는 파라메트릭과 비파라메트릭 메모리가 어떻게 상호작용하고 가장 효과적으로 결합하는지에 대한 새로운 연구 방향을 열어, 더 많은 제어 가능성과 해석 가능성을 제공하는 실제 사실 지식에 강하게 기반한 적용에 가능성을 보여줍니다.

---

## 📚 참고 자료 및 출처

| 번호 | 제목 / 출처 | 링크 |
|---|---|---|
| 1 | **[원논문]** Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020 | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| 2 | **[원논문 PDF]** NeurIPS 2020 공식 논문 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) |
| 3 | **[ACM]** Retrieval-augmented generation for knowledge-intensive NLP tasks, ACM Digital Library | [dl.acm.org](https://dl.acm.org/doi/abs/10.5555/3495724.3496517) |
| 4 | **[Meta AI]** Meta AI Research Publications | [ai.meta.com](https://ai.meta.com/research/publications/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks/) |
| 5 | **[Semantic Scholar]** RAG 논문 인용 분석 | [semanticscholar.org](https://www.semanticscholar.org/paper/Retrieval-Augmented-Generation-for-NLP-Tasks-Lewis-Perez/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31) |
| 6 | **[서베이]** "A Systematic Review of Key RAG Systems: Progress, Gaps, and Future Directions," arXiv 2025 | [arxiv.org/html/2507.18910](https://arxiv.org/html/2507.18910v1) |
| 7 | **[서베이]** "A Comprehensive Survey of RAG: Evolution," arXiv 2024 | [arxiv.org/pdf/2410.12837](https://arxiv.org/pdf/2410.12837) |
| 8 | **[서베이]** "RAG: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers," arXiv 2025 | [arxiv.org/html/2506.00054](https://arxiv.org/html/2506.00054v1) |
| 9 | **[도메인 적응]** "Improving the Domain Adaptation of RAG Models for Open Domain QA," TACL 2023 | [direct.mit.edu](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00530/114590/) |
| 10 | **[GraphRAG 서베이]** "Graph Retrieval-Augmented Generation: A Survey," ACM TOIS | [dl.acm.org](https://dl.acm.org/doi/10.1145/3777378) |
| 11 | **[HippoRAG 2]** "From RAG to Memory: Non-Parametric Continual Learning for LLMs," ICML 2025 | [arxiv.org/abs/2502.14802](https://arxiv.org/abs/2502.14802) |
| 12 | **[환각 완화]** "Hallucination Mitigation for Retrieval-Augmented LLMs: A Review," Mathematics 2025 | [mdpi.com](https://www.mdpi.com/2227-7390/13/5/856) |
| 13 | **[RAFT]** "RAFT: Adapting Language Model to Domain Specific RAG," 2024 | [shishirpatil.github.io](https://shishirpatil.github.io/publications/raft-2024.pdf) |
| 14 | **[블로그 분석]** "RAG Lewis 2020 Paper: Understanding the Original RAG Research," Latenode Blog, 2026 | [latenode.com](https://latenode.com/blog/ai-frameworks-technical-infrastructure/rag-retrieval-augmented-generation/rag-lewis-2020-paper-understanding-original-retrieval-augmented-generation-research) |
| 15 | **[리뷰]** NumByNum: Detailed Paper Review of RAG (Lewis et al., 2020), Medium | [medium.com](https://medium.com/@AriaLeeNotAriel/numbynum-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-lewis-et-al-df93a0f4c8f0) |
| 16 | **[Prompt Engineering Guide]** RAG for LLMs Overview | [promptingguide.ai](https://www.promptingguide.ai/research/rag) |

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., 2020)는 사전학습된 매개변수 기억(parametric memory)과 비매개변수 기억(non-parametric memory)을 결합하는 일반적인 미세조정 방법을 제시합니다. 이 논문의 핵심 주장은 **대규모 언어모델의 고정된 지식만으로는 지식집약적 작업에서 성능이 제한되며, 외부 지식 인덱스를 동적으로 접근할 수 있도록 하면 사실성과 구체성이 크게 향상된다**는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba741c31-27e1-4662-8123-fbe45bf68617/2005.11401v4.pdf)

주요 기여는 다음과 같습니다:

- **통합 RAG 아키텍처**: BART 생성 모델과 DPR 밀집 검색기, Wikipedia 벡터 인덱스를 end-to-end로 학습하는 프레임워크 제시
- **두 가지 RAG 공식화**: RAG-Sequence(동일 문서로 전체 시퀀스 생성)와 RAG-Token(토큰마다 다른 문서 사용)
- **상태-최고-기술 성능**: 3개의 오픈도메인 QA 작업에서 새로운 최고 성능 달성
- **생성 품질 향상**: BART 대비 더 구체적이고 다양하며 사실적인 텍스트 생성

***

## 2. 문제 정의, 방법론, 및 모델 구조

### 2.1 해결하는 문제

기존 사전학습 언어모델의 한계: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba741c31-27e1-4662-8123-fbe45bf68617/2005.11401v4.pdf)

1. **지식 접근성 제한**: 매개변수에 암묵적으로 저장된 지식은 정확한 조작이 어려움
2. **할루시네이션**: 존재하지 않거나 부정확한 정보 생성
3. **지식 업데이트 불가**: 세상이 변해도 모델 재학습 필요
4. **해석 불가**: 생성된 텍스트의 출처 추적 어려움

지식집약적 작업(knowledge-intensive tasks)은 인간이 외부 자료 없이 수행할 수 없는 작업으로, 기존 seq2seq 모델보다 특화된 아키텍처가 필요했습니다.

### 2.2 제안 방법 및 수학적 공식

#### RAG-Sequence 모델

$$p_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-k}(p(\cdot|x))} p_\eta(z|x) p_\theta(y|x, z)$$

여기서:
$$p_\theta(y|x, z) = \prod_{i=1}^{N} p_\theta(y_i|x, z, y_{1:i-1})$$

- 입력 $x$에 대해 상위 K개 문서를 검색
- 각 문서 $z$에 대해 생성 확률 $p_\theta(y|x,z)$ 계산
- 모든 문서 확률을 $p_\eta(z|x)$로 가중 평균하여 최종 출력 확률 계산

**특징**: 모든 출력 토큰에 대해 동일한 문서를 사용

#### RAG-Token 모델

$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-k}(p(\cdot|x))} p_\eta(z|x) p_\theta(y_i|x, z, y_{1:i-1})$$

**특징**: 각 토큰마다 다른 문서를 선택 가능 → 여러 문서의 정보 조합

#### 검색기: DPR (Dense Passage Retriever)

$$p_\eta(z|x) \propto \exp(d(z)^\top q(x))$$

여기서:
- $d(z) = \text{BERT}_d(z)$ : 문서 인코더 (BERT-base)
- $q(x) = \text{BERT}_q(x)$ : 쿼리 인코더 (BERT-base)
- 최대 내적 검색(MIPS)으로 top-K 문서 추출

각 문서는 100단어 청크로 분할되어 21M개 문서 인덱스 생성

#### 생성기: BART

$$p_\theta(y_i|x, z, y_{1:i-1})$$

- BART-large (400M 파라미터) seq2seq 트랜스포머
- 입력 $x$와 검색된 문서 $z$를 연결하여 입력
- 자동회귀식 생성

### 2.3 모델 구조

| 구성요소 | 설명 | 파라미터 |
|---------|------|---------|
| **검색기** | DPR 양인코더 아키텍처 | 110M (고정) |
| **문서 인덱스** | Wikipedia 밀집 벡터 | 21M × 768차원 |
| **생성기** | BART-large seq2seq | 406M (학습) |
| **총 학습가능 파라미터** | 쿼리 인코더 + BART | 516M |

**훈련 방식**:
- 검색된 문서를 잠재변수로 취급
- 음의 한계 로그 우도(negative marginal log-likelihood) 최소화
- 문서 인코더는 고정, 쿼리 인코더와 BART만 학습
- 이유: 문서 인덱스 업데이트 계산 비용 높음

**디코딩 방식**:
- **RAG-Token**: 표준 빔서치 사용 (토큰별 확률 분해 가능)
- **RAG-Sequence**: "철저한 디코딩" 또는 "빠른 디코딩" 옵션
  - 각 문서별로 빔서치 실행 후 확률 합산

***

## 3. 성능 향상 및 한계

### 3.1 벤치마크 성능 비교

#### 오픈도메인 질문 답변 (표 1)

| 모델 | Natural Questions | TriviaQA | WebQuestions | CuratedTrec |
|-----|-------------------|----------|-------------|------------|
| T5-11B | 34.5% | 50.1% | 37.4% | - |
| REALM | 40.4% | - | 40.7% | 46.8% |
| **DPR** | 41.5% | 57.9% | 41.1% | 50.6% |
| **RAG-Token** | 44.1% | 55.2% | 45.5% | 50.0% |
| **RAG-Seq** | **44.5%** | **56.8%** | **45.2%** | **52.2%** |

**주요 성과**:
- NQ에서 DPR 대비 **+3.0%** (44.5% vs 41.5%)
- 추출형 모델보다 생성형 답변이 우수
- 문서에 답이 없는 경우 11.8% 정확도 달성 (추출형은 0%)

#### 생성 작업 성능

| 작업 | 메트릭 | BART | RAG-Token | RAG-Seq |
|-----|--------|------|-----------|---------|
| MS-MARCO | BLEU-1 | 41.6 | 41.5 | **44.2** |
| MS-MARCO | ROUGE-L | 38.2 | 40.1 | **40.8** |
| Jeopardy | Q-BLEU-1 | 19.7 | **22.2** | 21.4 |
| FEVER-3 | Acc | 64.0 | 72.5 | **72.5%** |

**인간 평가 결과** (Jeopardy):
- RAG가 더 사실적: 42.7% vs BART 7.1%
- RAG가 더 구체적: 37.4% vs BART 16.8%

### 3.2 생성 품질 분석

**다양성** (표 5):
- RAG-Seq 삼중글램 다양성: 83.5%
- RAG-Token: 77.8%
- BART: 70.7%
- 금 데이터: 89.6%

→ RAG는 다양성 증대 디코딩 없이도 BART보다 자연스럽게 다양한 생성

### 3.3 학습 가능한 검색의 효과 (표 6)

| 모델 설정 | NQ EM | Jeopardy Q-BLEU | MS-MARCO BLEU |
|---------|--------|-----------------|-----------------|
| RAG-Seq (학습 안함) | 41.2% | 11.8% | 19.6% |
| **RAG-Seq (학습)** | **44.0%** | **15.3%** | **21.5%** |
| **개선도** | **+2.8%** | **+3.5%** | **+1.9%** |

학습 가능한 검색은 모든 작업에서 성능 향상

### 3.4 주요 한계

#### a) 검색 붕괴 (Retrieval Collapse)

>일부 작업(예: 스토리 생성)에서 검색기가 입력과 관계없이 동일한 문서만 검색하는 문제 발생. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba741c31-27e1-4662-8123-fbe45bf68617/2005.11401v4.pdf)

원인:
- 긴 타겟 시퀀스에서 약한 그래디언트
- 덜 명확한 사실 지식 필요

#### b) 검색기-생성기 불일치

- 검색기는 답을 포함한 문서 검색 최적화
- 생성기는 맥락에서 답 생성 최적화
- 이 둘의 목표가 완벽히 일치하지 않음

#### c) 계산 복잡도

- RAG-Sequence의 철저한 디코딩: 여러 forward pass 필요
- 빠른 디코딩: 근사로 인한 성능 손실

#### d) 도메인 특화성

- Wikipedia 기반 사전학습 검색기는 특화 도메인에 최적화 안 됨
- 검색 지도 신호 없이 빠른 적응 어려움

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 현재 논문의 일반화 분석

논문에서는 제한적인 일반화 분석만 제시:

#### 인덱스 핫스왑 실험 (Index Hot-Swapping)

RAG의 비매개변수 메모리 교체를 통한 지식 업데이트 능력 입증:

| 설정 | 세계 지도자 질문 (2016) | 세계 지도자 질문 (2018) |
|-----|----------------------|----------------------|
| 2016 인덱스 사용 | 70% | 12% |
| 2018 인덱스 사용 | 4% | 68% |

→ **인덱스 교체만으로 모델 재학습 없이 지식 업데이트 가능**

이는 매개변수 기억만의 모델과 달리 RAG가 근본적으로 더 나은 확장성을 가짐을 보여줍니다.

### 4.2 2020년 이후 일반화 성능 연구 (주요 개선사항)

#### (1) REALM (Guu et al., 2020)

$$p(y|x) = \sum_z p(z|x) p(y|x,z)$$

**RAG와의 차이**:
- 사전학습 단계에서 검색기와 생성기를 동시 학습
- 마스크 언어 모델링을 학습 신호로 사용
- 결과: REALM이 특정 작업에서 더 강한 일반화

#### (2) 도메인 적응 RAG (2022)

**RAG-end2end** (MIT-CSAIL, 2022): [direct.mit](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00530/114590/Improving-the-Domain-Adaptation-of-Retrieval)

- 문서 인코더도 학습 가능하게 변경
- 도메인 특화 보조 신호 주입
- **결과**: COVID-19, News, Conversation 도메인에서 기존 RAG 대비 유의미한 개선

| 도메인 | 기존 RAG | RAG-end2end |
|--------|---------|-----------|
| COVID-19 | - | ✓ 개선 |
| News | - | ✓ 개선 |
| Conversations | - | ✓ 개선 |

#### (3) 다중 쿼리 생성을 통한 일반화 개선

**문제**: 단일 쿼리-문서 쌍만으로 훈련하면 과적합 발생

**해결책** (Rajapakse, 2023): [dl.acm](https://dl.acm.org/doi/10.1007/978-3-031-28238-6_7)

- 동일 문서에 대해 여러 쿼리 생성
- DPR 재훈련

**결과**:
- 도메인 외 테스트셋에서 **5-10% 성능 향상**
- 특히 제로샷 설정에서 강한 일반화

#### (4) 시퀀스 수준 증류를 통한 다중 도메인 일반화

**연구** (NAVER Labs, 2025): [arxiv](https://arxiv.org/pdf/2504.02411.pdf)

표준 미세조정: 도메인 외 일반화 실패

**시퀀스 수준 증류** (teacher-generated labels):
- 50%+ 도메인 외 성능 개선
- 더 일관성 있는 감독 신호 제공

#### (5) 적응형 검색 메커니즘

**주요 개선안들**:

| 방법 | 핵심 아이디어 | 일반화 효과 |
|-----|------------|----------|
| **Self-CRAG** (2024) | 자기 비판적 RAG | PopQA에서 320% 개선 |
| **ARM-RAG** (2023) | 보조 근거 메모리 | 수학 문제 성능 향상 |
| **WebFilter** (2025) | 웹 검색 도구 학습 | 도메인 외 Musique 30% 정확도 |

### 4.3 일반화 성능을 저해하는 요소

#### (1) 밀집 검색기의 발견된 편향 (2025)

**연구** (Fayyaz et al., 2025): [arxiv](https://arxiv.org/pdf/2503.05037.pdf)

밀집 검색기들이 다음에 편향:
- **길이 편향**: 짧은 문서 선호 (정확도 손상)
- **리터럴 편향**: 정확한 단어 일치 선호 (의미적 변형 미탐지)
- **위치 편향**: 문서 앞의 엔티티 선호

→ 이런 휴리스틱 편향이 도메인 외 성능 저하의 주요 원인

#### (2) 검색-생성 미스매치

**문제**:
- 검색 단위 입자도(granularity) 부적절
- 문서와 생성 요구사항 불일치

**해결책** (2023-2024):
- 제안 수준 검색: 문장보다 작은 단위
- 쿼리 확장: 의미적 임베딩으로 세밀화

***

## 5. 논문의 미래 연구에 미치는 영향 및 고려사항

### 5.1 RAG 연구에 미친 직접적 영향

#### 인용도 및 영향력

- **Google Scholar**: 14,893회 인용 (2024년 기준) [arxiv](https://arxiv.org/abs/2005.11401)
- 이는 2020년 이후 NLP 분야에서 가장 영향력 있는 논문 중 하나

#### RAG 패러다임의 확산

1. **학술 연구**: 수백 편의 후속 연구
2. **산업 적용**: 
   - OpenAI ChatGPT-4 (검색 기능)
   - Google AI 제품들
   - 의료, 법률, 금융 도메인 응용

3. **오픈소스 생태계**:
   - HuggingFace Transformers 통합
   - RAG Foundry, FlashRAG 등 프레임워크 개발

### 5.2 향후 연구 시 고려할 점

#### A. 아키텍처 차원

| 현안 | 제안하는 방향 |
|------|------------|
| **검색-생성 불일치** | 검색기와 생성기를 end-to-end로 공동 최적화 |
| **매개변수 크기** | RAG는 T5-11B보다 훨씬 소형 (516M)이면서 경쟁력 있음 |
| **검색 입자도** | 문서 청크 크기 최적화 (문장 vs 제안 vs 단락) |

#### B. 훈련 차원

**핵심 개선안**:

1. **다중 쿼리 훈련**
   - 단일 문서-쿼리 쌍 편향 극복
   - 도메인 외 일반화 5-10% 향상

2. **시퀀스 수준 증류**
   - 표준 미세조정보다 도메인 외 성능 50% 이상 향상 [arxiv](https://arxiv.org/pdf/2504.02411.pdf)
   - 더 일관성 있는 감독 신호

3. **하드 네거티브 마이닝**
   - ANCE (Xiong et al., 2020)
   - 검색기 판별력 향상

#### C. 도메인 적응

| 전략 | 효과 | 비용 |
|------|------|------|
| 인덱스 핫스왑 | 낮음 | 매우 낮음 |
| 검색기 미세조정 | 중간 | 낮음 |
| 생성기 미세조정 | 높음 | 중간 |
| 전체 end-to-end 재학습 | 최고 | 높음 |

#### D. 평가 메트릭

**한계**:
- 기존 BLEU, ROUGE는 검색 기반 생성 품질 미측정
- 필요: **검색 인식 평가 메트릭**

**개발 필요**:
- 사실성 검증 자동화
- 출처 속성 점수
- 다중홉 추론 정확도

#### E. 견고성 강화

**밝혀진 7가지 실패 지점** (Barnett et al., 2024): [arxiv](https://arxiv.org/abs/2401.05856)

1. 검색 실패: 관련 문서 미검색
2. 컨텍스트 누락: 문서 검색되었으나 맥락 미포함
3. 통합 실패: 많은 문서에서 답 선택 실패
4. 추출 실패: 답이 있으나 모델이 미추출
5. 추론 실패: 다중홉 추론 불가
6. 지식 충돌: 매개변수 vs 검색된 지식 모순
7. 생성 실패: 문법 또는 연역 오류

**대응 방안**:
- 각 단계별 진단 자동화
- 반복적 검색 및 검증
- 모순 해결 메커니즘

#### F. 실시간 및 동적 지식 통합

**원본 RAG의 한계**:
- 정적 인덱스 (학습 후 변경 불가)
- 최신 정보 통합 불가

**향후 방향**:
- 스트리밍 인덱스 업데이트
- 온라인 학습 가능 검색기
- 시간-인식 검색 (temporal-aware)

#### G. 프라이버시 보호

**고려사항**:
- 개인정보 포함 코퍼스 안전한 인덱싱
- 연합 검색 (federated retrieval)
- 차등 프라이버시 기법 통합

### 5.3 2020년 이후 최신 연구 비교 분석

#### 주요 진전 타임라인

| 시기 | 주요 연구 | 기여 |
|------|---------|------|
| **2020** | REALM, RAG | 기본 프레임워크 확립 |
| **2020-2021** | DPR, ColBERT, RETRO | 검색 메커니즘 고도화 |
| **2021-2022** | ColBERTv2, RAG 도메인 적응 | 압축, 효율성, 특화 도메인 |
| **2022-2023** | 다중도메인 RAG, 오류 분석 | 견고성, 일반화 연구 |
| **2023-2024** | 적응형 검색, 혼합 전문가 | 토큰별 최적화, 모듈식 설계 |
| **2024-2025** | OpenRAG, 구조화된 RAG | End-to-end 최적화, 이질 소스 |

#### 성능 비교 (NQ 벤치마크)

```
2020: RAG-Seq     44.5%  ← 원본 논문
2021: DPR         41.5%
2021: REALM       40.4%
2022: ColBERT     ≈ 45%
2023: RAG-end2end 46%+   (도메인 적응)
2024: Self-CRAG   ≈ 48%+ (자기 비판)
2025: OpenRAG     ≈ 49%+ (in-context 학습)
```

#### 검색기 진화

| 방법 | 아키텍처 | 장점 | 한계 |
|------|---------|------|------|
| **BM25** | 단어 중첩 | 해석 가능, 빠름 | 의미 격차 |
| **DPR** | 단일 벡터 | 밀집, 확장 가능 | 과적합, 편향 |
| **ColBERT** | 다중 벡터 | 세밀한 상호작용 | 저장 용량 큼 |
| **ColBERTv2** | 압축 + 증류 | 최고 품질, 저용량 | 훈련 복잡 |
| **LLM-기반** | LLM 인코더 | 의미 이해 강함 | 느림 |

#### 생성기 진화

| 모델 | 크기 | 일반화 | 속도 |
|------|------|--------|------|
| **BART** | 400M | 중간 | 빠름 |
| **T5** | 11B | 우수 | 느림 |
| **LLaMA** | 7-70B | 강함 | 중간 |
| **GPT-4** | ? | 최고 | 최고 비용 |

#### 도메인 특화 성과

**의료 도메인**:
- PubMedQA에서 BioBERT 기반 검색기 효과적
- ColBERT 재순위화는 추가 이득 없음 (효율성 우수)

**금융 도메인**:
- 특화 인덱스 + 도메인 미세조정: 일반 모델 대비 20-30% 향상

**법률 도메인**:
- 구조화된 지식 그래프 + RAG: 판례 관련성 90%+

***

## 결론

**RAG는 2020년 이후 NLP의 가장 중요한 패러다임 전환 중 하나**입니다. 원본 논문의 핵심 통찰—매개변수와 비매개변수 메모리의 결합—은 여전히 유효하며, 다음 세대 연구가 이를 기반으로 견고성, 일반화, 효율성을 개선하고 있습니다.

**특히 주목할 점**:
1. **일반화 문제 해결의 필수성**: 표준 미세조정의 도메인 외 성능 저하는 심각하며, 시퀀스 수준 증류 등의 기법이 50%+ 개선 가능
2. **검색-생성 공동 최적화의 중요성**: end-to-end 학습과 다중 쿼리 훈련이 향후 핵심
3. **모듈식 설계의 강점**: 인덱스 교체, 검색기 업그레이드 등이 별도 재학습 없이 가능
4. **도메인 적응의 실질화**: RAG-end2end, SimRAG 등으로 특화 도메인 고성능 달성 가능

따라서 지식집약적 작업을 위한 미래 LLM 구축 시 **RAG 기반의 하이브리드 아키텍처는 필수 고려 대상**입니다.

***

## 참고자료 (References)

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76]</span>

<div align="center">⁂</div>

[^1_1]: 2005.11401v4.pdf

[^1_2]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00530/114590/Improving-the-Domain-Adaptation-of-Retrieval

[^1_3]: https://dl.acm.org/doi/10.1007/978-3-031-28238-6_7

[^1_4]: https://arxiv.org/pdf/2504.02411.pdf

[^1_5]: https://arxiv.org/pdf/2503.05037.pdf

[^1_6]: https://arxiv.org/abs/2005.11401

[^1_7]: https://arxiv.org/abs/2401.05856

[^1_8]: https://arxiv.org/html/2506.00054v1

[^1_9]: https://aclanthology.org/2022.naacl-main.272.pdf

[^1_10]: https://aclanthology.org/2022.naacl-srw.7

[^1_11]: https://arxiv.org/abs/2203.16714

[^1_12]: https://arxiv.org/abs/2210.02928

[^1_13]: https://aclanthology.org/2022.acl-long.579

[^1_14]: https://www.semanticscholar.org/paper/0768cacd594fe087a6187c5464770c3af6b66ee7

[^1_15]: https://arxiv.org/abs/2205.12230

[^1_16]: https://arxiv.org/abs/2503.13281

[^1_17]: https://arxiv.org/abs/2311.04177

[^1_18]: https://ascopubs.org/doi/10.1200/JCO.2024.42.16_suppl.e13637

[^1_19]: https://arxiv.org/pdf/2503.08398.pdf

[^1_20]: http://arxiv.org/pdf/2502.14614.pdf

[^1_21]: https://arxiv.org/pdf/2405.13576.pdf

[^1_22]: https://arxiv.org/html/2504.06271v1

[^1_23]: http://arxiv.org/pdf/2410.05801.pdf

[^1_24]: https://arxiv.org/html/2406.00944v1

[^1_25]: https://arxiv.org/pdf/2406.05085v1.pdf

[^1_26]: https://arxiv.org/html/2405.00175

[^1_27]: http://proceedings.mlr.press/v119/guu20a/guu20a.pdf

[^1_28]: https://arxiv.org/html/2507.18910v1

[^1_29]: https://www.emergentmind.com/topics/dense-passage-retriever-dpr

[^1_30]: https://www.youtube.com/watch?v=lj-LGrnh1oU

[^1_31]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00564/116466/Questions-Are-All-You-Need-to-Train-a-Dense

[^1_32]: https://arxiv.org/abs/2002.08909

[^1_33]: https://www.promptingguide.ai/kr/research/rag

[^1_34]: https://chentong0.github.io/factoid-wiki/

[^1_35]: https://arxiv.org/pdf/2002.08909.pdf

[^1_36]: https://www.sciencedirect.com/science/article/pii/S2666920X25000578

[^1_37]: https://openreview.net/pdf/a5931b87a067e19d26e4bb6f67af929ead3d822d.pdf

[^1_38]: https://dl.acm.org/doi/abs/10.5555/3524938.3525306

[^1_39]: https://arxiv.org/html/2504.05216v1

[^1_40]: https://arxiv.org/html/2509.10697v1

[^1_41]: https://arxiv.org/html/2410.15801v1

[^1_42]: https://www.semanticscholar.org/paper/REALM:-Retrieval-Augmented-Language-Model-Guu-Lee/832fff14d2ed50eb7969c4c4b976c35776548f56

[^1_43]: https://arxiv.org/html/2402.11035v1

[^1_44]: https://arxiv.org/pdf/2410.12837.pdf

[^1_45]: https://arxiv.org/html/2312.16821v1

[^1_46]: https://arxiv.org/html/2407.19813v3

[^1_47]: https://arxiv.org/html/2312.06648v1

[^1_48]: https://arxiv.org/pdf/2308.09308.pdf

[^1_49]: https://www.semanticscholar.org/paper/Improving-the-Generalizability-of-the-Dense-Passage-Rajapakse-Rijke/147a8393b739b0cfa70b626bc69c90782c896592

[^1_50]: https://arxiv.org/pdf/2408.02545.pdf

[^1_51]: https://arxiv.org/pdf/2412.13018.pdf

[^1_52]: https://arxiv.org/pdf/2410.17952.pdf

[^1_53]: https://arxiv.org/pdf/2402.17497v2.pdf

[^1_54]: http://arxiv.org/pdf/2411.08438.pdf

[^1_55]: https://europe.naverlabs.com/research/publications/adapting-large-language-models-for-multi-domain-retrieval-augmented-generation/

[^1_56]: https://unfoldai.com/rag-limitations/

[^1_57]: https://www.leximancer.com/blog/everything-wrong-with-retrieval-augmented-generation

[^1_58]: https://pangyoalto.com/en/colbertv1-2-review-en/

[^1_59]: https://aclanthology.org/2024.emnlp-main.1236.pdf

[^1_60]: https://www.chitika.com/rag-challenges-and-solution/

[^1_61]: https://arxiv.org/abs/2201.08471

[^1_62]: https://arxiv.org/pdf/2406.05654.pdf

[^1_63]: https://www.merge.dev/blog/rag-challenges

[^1_64]: https://www.arxiv.org/pdf/2509.23861.pdf

[^1_65]: https://openreview.net/forum?id=ZS4m74kZpH

[^1_66]: https://arxiv.org/html/2401.05856v1

[^1_67]: https://aclanthology.org/2025.acl-long.447.pdf

[^1_68]: https://arxiv.org/pdf/2401.05856.pdf

[^1_69]: https://arxiv.org/html/2508.07956v1

[^1_70]: https://arxiv.org/html/2507.02962v6

[^1_71]: https://arxiv.org/abs/2506.00054

[^1_72]: https://arxiv.org/html/2510.01612v3

[^1_73]: https://arxiv.org/pdf/2510.00552.pdf

[^1_74]: https://arxiv.org/html/2502.20245v1

[^1_75]: https://arxiv.org/html/2510.15191v1

[^1_76]: https://arxiv.org/pdf/2504.08744.pdf
