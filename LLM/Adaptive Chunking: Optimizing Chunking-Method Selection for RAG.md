# Adaptive Chunking: Optimizing Chunking-Method Selection for RAG 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 RAG 시스템의 "one-size-fits-all" 청킹 전략은 다양한 구조와 의미를 가진 문서에 적용 시 필연적으로 성능 저하를 야기한다. 논문은 **문서별로 최적의 청킹 방법을 동적으로 선택**하는 **Adaptive Chunking** 프레임워크를 제안하며, 이를 위한 **5가지 내재적(intrinsic) 평가 지표**를 새롭게 정의한다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| ① 내재적 평가 지표 5종 | RC, ICC, DCC, BI, SC — 청킹 품질을 다운스트림 성능 없이 직접 평가 |
| ② 신규 청킹 기법 2종 | LLM-Regex Splitter, Split-then-Merge Recursive Splitter + 후처리 기법 |
| ③ 평가 파이프라인 | 검색 품질 + 생성 정확도를 동시에 측정하는 새로운 평가 체계 |

**성과 요약:**
- 정답 정확도: 62–64% → **72%** 향상
- 답변 가능 질문 수: 49 → **65개** (+32.7%)
- 모델, 프롬프트 변경 없이 청킹만으로 달성

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**Context-Preservation Dilemma:** 기존 청킹 방법들은 문서의 자연스러운 구조와 의미론적 경계를 무시하고 임의의 경계를 부과하여 다음과 같은 문제를 발생시킨다:

1. **논리적 단위 분절** — 단락, 표, 제목-본문 쌍이 끊김 (Duarte et al., 2024)
2. **크기 불일치** — 너무 크거나 작은 청크가 임베딩 품질 저하 (Günther et al., 2025)
3. **주제 혼재** — 무관한 내용이 하나의 청크에 묶임 (Qu et al., 2025)
4. **평가 기준 부재** — 대부분 연구가 Hits@k, nDCG@k 같은 다운스트림 지표로만 청킹을 간접 평가

기존 방법들의 한계를 정리하면:

| 방법 | 장점 | 단점 |
|------|------|------|
| Sentence-based | 의미 응집력 | 논리 블록 파괴 |
| Recursive (LangChain) | 길이 제어 | 무관 주제 혼재 |
| Semantic Chunker | 의미 경계 탐지 | 높은 계산 비용 |
| LLM-driven | 자연 경계 인식 | 단일 전략 강제, 높은 오버헤드 |

---

### 2-2. 제안 방법 및 수식

#### 이상적인 청크의 4가지 속성

1. **자기완결성(Self-contained)** — 독립적으로 이해 가능
2. **길이 제약 준수(Length compliance)** — 임베딩 모델 토큰 한계 고려
3. **의미적 응집성(Semantic cohesion)** — 단일 주제 집중
4. **컨텍스트 보존(Context-preserving)** — 문서 자연 구조 반영

---

#### 📐 5가지 내재적 평가 지표 (수식 포함)

**① References Completeness (RC)**

개체-대명사 쌍이 청크 경계에 의해 분리되지 않는 비율. Maverick 공참조 해소 모델 사용.

$P = \{(e_i, p_i)\}_{i=1}^{N}$: 개체-대명사 쌍 집합 ( $s_i = \text{start}(e_i)$ , $t_i = \text{end}(p_i)$ )

$B$: 내부 청크 경계 집합

$$m_i = \mathbf{1}\left[\exists\, b \in B : s_i < b \leq t_i\right]$$

$$\text{RC} = 1 - \frac{1}{N}\sum_{i=1}^{N} m_i$$

> $m_i = 1$이면 경계가 개체와 대명사 사이를 가르는 경우 (참조 손실)

---

**② Block Integrity (BI)**

단락, 표, 그림, 제목-본문 쌍 등 구조 단위가 온전히 유지되는 비율. 허용 오차 $\tau = 5$ 문자.

$G = \{0, d_1, \ldots, d_M, L\}$: 파서가 제공한 블록 경계

$$I_j = \mathbf{1}\left[\nexists\, b \in B : d_j + \tau < b < d_{j+1} - \tau\right]$$

$$\text{BI} = \frac{1}{|G|-1}\sum_{j=0}^{|G|-1} I_j$$

---

**③ Intrachunk Cohesion (ICC)**

청크 내 문장 임베딩과 전체 청크 임베딩 간 코사인 유사도의 평균. `jina-embeddings-v3` 사용.

$\mathbf{v}(c_k) \in \mathbb{R}^d$: 청크 $c_k$의 정규화된 임베딩

$\mathbf{v}(s_{kj})$: 청크 내 $j$번째 문장의 정규화된 임베딩

$$\text{Cohesion}(c_k) = \frac{1}{n_k}\sum_{j=1}^{n_k} \mathbf{v}(s_{kj})^\top \mathbf{v}(c_k), \quad n_k \geq 2$$

```math
\text{ICC} = \max\left\{0,\, \frac{1}{|\mathcal{K}|}\sum_{k \in \mathcal{K}} \text{Cohesion}(c_k)\right\}
```

> $\mathcal{K}$: $n_k \geq 2$인 유효 청크 인덱스 집합

---

**④ Document Contextual Coherence (DCC)**

각 청크가 슬라이딩 윈도우(최대 $T = 3000$ 토큰) 내 문맥과 얼마나 잘 정렬되는지 측정.

$\mathbf{w}(W_m) \in \mathbb{R}^d$: 윈도우 $W_m$의 정규화된 임베딩

$$\text{Coherence}(W_m) = \frac{1}{|\mathcal{C}_m|}\sum_{k \in \mathcal{C}_m} \mathbf{w}(W_m)^\top \mathbf{v}(c_k)$$

```math
\text{DCC} = \max\left\{0,\, \frac{1}{|M|}\sum_{m=0}^{M} \text{Coherence}(W_m)\right\}
```

---

**⑤ Size Compliance (SC)**

청크 토큰 수가 사전 정의된 범위 $[m, M] = [100, 1100]$ 내에 있는 비율.

$\tau_k = \text{tok}(c_k)$: 청크 $c_k$의 토큰 수

$$\text{SC} = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}\left[m \leq \tau_k \leq M\right]$$

---

#### 신규 청킹 기법

**① LLM-Regex Splitter**

```
문서 앞부분(최대 8,000 토큰) 분석
→ LLM이 문서 구조에 맞는 정규식 패턴 생성
→ Python re.split()으로 전체 문서 분할
```

- 장점: LLM의 구조 분석 능력 + 정규식의 결정론적 일관성
- 적합 문서: 법률 문서처럼 명확한 구분자(조항 번호 등)가 있는 구조화 텍스트

**② Split-then-Merge Recursive Splitter**

```
1단계(Split): 우선순위 구분자 목록으로 재귀적 분할
  (제목 → 섹션 → 문장 → 문자)
  각 세그먼트 크기 ≤ S가 될 때까지

2단계(Merge): 인접 세그먼트를 S를 초과하지 않도록 탐욕적 병합
  → 오버랩 유지를 위해 백트래킹
  → 과대 청크는 재분할
```

**후처리 (Post-processing)**

| 단계 | 조건 | 처리 |
|------|------|------|
| Oversized-chunk splitting | > 1,100 토큰 | 동일 구분자 계층으로 재분할 |
| Tiny-chunk merging | < 100 토큰 | 인접 세그먼트와 병합 (최대 1,150 토큰) |

---

#### Adaptive Chunking 선택 메커니즘

각 문서에 대해 4가지 청킹 방법을 적용하고, 5개 지표의 **평균값**이 가장 높은 방법 선택:

$$\text{Best}(d) = \arg\max_{c \in \mathcal{C}} \frac{\text{RC}(d,c) + \text{ICC}(d,c) + \text{DCC}(d,c) + \text{BI}(d,c) + \text{SC}(d,c)}{5}$$

실험 결과 선택 분포:

| 선택된 방법 | 선택 비율 |
|-------------|-----------|
| page (post-processed) | 48% |
| our recursive (s=1100) | 42% |
| LLM regex (GPT-5) | 6% |
| our recursive (s=600) | 3% |

---

### 2-3. 모델 구조 및 파이프라인

#### 전체 RAG 파이프라인

```
문서 (PDF)
    ↓
[Azure AI Document Intelligence]
    ↓ Markdown 변환
[청킹 방법 선택: Adaptive Chunking]
    ↓
[하이브리드 검색]
  ├─ BM25 키워드 검색 (top-50)
  └─ Qwen3-Embedding-4B 의미 검색 (top-50)
    ↓
[Merge & Deduplicate] (100개 후보)
    ↓
[Reranking: snowflake-arctic-embed-l-v2.0] (top-10)
    ↓
[생성: GPT-4.1, T=0, top-p=1]
```

#### 사용 모델 목록

| 역할 | 모델 |
|------|------|
| 문서 파싱 | Microsoft Azure AI Document Intelligence |
| 임베딩 | Jina AI jina-embeddings-v3 |
| 쿼리 임베딩 | Qwen3-Embedding-4B |
| 공참조 해소 | Maverick (Sapienza NLP) |
| Reranker | snowflake-arctic-embed-l-v2.0 |
| LLM 생성 | GPT-4.1 (T=0) |
| 평가 (LLM-judge) | GPT-4.1 |

---

### 2-4. 성능 향상

#### 청킹 품질 지표 비교 (Table 3)

| 방법 | RC | ICC | DCC | BI | SC | Mean |
|------|-----|-----|-----|-----|-----|------|
| Adaptive Chunking | **99.0** | 68.2 | 88.8 | **99.4** | **99.9** | **91.07** |
| our recursive (s=1100) | **99.0** | 66.6 | **89.7** | 98.1 | **100.0** | 90.68 |
| page (post-processed) | 97.2 | 69.2 | 86.4 | **99.9** | 99.9 | 90.52 |
| LC recursive (default) | 96.1 | 65.6 | 88.8 | 95.0 | 97.7 | 88.62 |
| sentence | 86.3 | **78.4** | 72.5 | 61.9 | 67.2 | 73.26 |

> 통계적 유의성: Wilcoxon 부호순위 검정, $p < 0.001$

#### RAG 다운스트림 성능 비교 (Table 5)

| 지표 | Adaptive Chunking | LC recursive (default) | page (raw) |
|------|------------------|----------------------|-----------|
| Retrieval Completeness | **67.68%** | 58.08% | 59.09% |
| Answer Correctness | **78.01%** | 70.11% | 73.33% |
| Mean | **71.77%** | 62.07% | 63.80% |
| Answered queries | **65/99** | 49/99 | 49/99 |

**핵심 발견: 증폭 효과 (Amplification Effect)**

내재적 지표 차이는 0.4–2.4%p에 불과하지만, RAG 성능 격차는 8–10%p(검색)~5–8%p(생성)로 크게 확대됨. 이는 청크 품질 개선이 검색-생성 파이프라인을 통해 복합적으로 증폭됨을 시사.

---

### 2-5. 한계점

| 한계 | 상세 내용 |
|------|----------|
| **계산 효율성** | DCC 계산(15:58)과 공참조 추출(13:13)이 평가 시간의 대부분 차지 |
| **언어 제약** | Maverick 모델이 영어 전용 → RC 지표의 다국어 적용 불가 |
| **도메인 제약** | 공식 문서(법률, 기술, 사회과학)에 한정; 비공식 텍스트, 창작물, 멀티모달 문서 미검증 |
| **하이퍼파라미터** | 청크 크기(100–1,100 토큰), 슬라이딩 윈도우(3,000 토큰), 지표 가중치(동일) 모두 휴리스틱 |
| **쿼리 무관 선택** | 인덱싱 시점에 방법을 고정; 쿼리 특성이나 태스크에 따른 동적 조정 불가 |
| **블록 무결성 의존성** | BI 계산이 파서 제공 블록 스팬 어노테이션 필요 |
| **코퍼스 규모** | 33개 문서로 한정된 실험 규모 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화를 지원하는 설계 요소

**① 문서 유형 독립적 메트릭 설계**

5가지 지표는 특정 도메인 지식 없이도 적용 가능한 구조적·의미적 속성을 측정한다. 법률 문서(16개), 기술 문서(9개), 사회과학 문서(8개)에서 동일한 지표가 일관된 성능 개선을 보였으며, 추가 데이터셋에서도 일관된 성과를 확인했다고 보고:

> *"While we report detailed results on one corpus for reproducibility, we validated our approach on additional datasets with consistent performance gains."*

**② Adaptive Selection의 범용성**

단일 고정 전략 대신 문서별 동적 선택은 새로운 도메인/문서 타입에도 자연스럽게 확장 가능. 선택 풀(candidate pool)에 새로운 청킹 방법을 추가하면 자동으로 경쟁에 포함된다.

**③ 지표 간 상보성(Complementarity)**

Spearman 상관이 $-0.44 < \rho < 0.31$로 낮아, 5개 지표가 서로 독립적인 청킹 속성을 포착함을 의미. 다양한 문서 특성에 대해 편향 없는 평가 가능.

$$-0.44 < \rho(\text{metric}_i, \text{metric}_j) < 0.31$$

**④ 후처리의 범용적 효과**

포스트프로세싱이 LLM-Regex(SC: 58.3% → 99.6%), Semantic Chunker(SC: 48.1% → 99.9%) 등 다양한 방법에서 큰 개선을 보임 → 청킹 방법 종류와 무관하게 일반화 가능.

### 3-2. 일반화 한계 및 개선 방향

| 한계 요인 | 현재 상태 | 개선 방향 |
|----------|----------|----------|
| 언어 다양성 | 영어 전용 (RC 지표) | 다국어 공참조 모델 통합 |
| 문서 형식 | Markdown 기반 | PDF, HTML, LaTeX 등 다양한 파서 지원 |
| 지표 가중치 | 균등 가중 | 도메인별 학습 기반 가중치 최적화 |
| 청크 크기 하이퍼파라미터 | 고정 (100–1,100) | 임베딩 모델별 자동 조정 |
| 비공식 텍스트 | 미검증 | SNS, 대화체, 창작 문서 추가 실험 |

### 3-3. 일반화를 위한 추가 연구 제언

1. **메타학습(Meta-learning) 기반 방법 선택**: 문서 특성(도메인, 길이, 구조 복잡도)을 입력으로 받아 최적 청킹 방법을 예측하는 경량 분류기 학습

2. **지표 가중치 자동화**: 도메인별로 어떤 지표가 다운스트림 성능과 더 강한 상관을 보이는지 분석하여 가중치를 동적으로 설정

3. **온라인 적응(Online Adaptation)**: 쿼리 피드백을 활용해 인덱싱 이후에도 청킹 전략을 점진적으로 조정

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 청킹 방법론 관련 연구

| 논문 | 연도 | 방법 | Adaptive Chunking과의 비교 |
|------|------|------|--------------------------|
| **Stanza** (Qi et al.) | 2020 | 문장 기반 NLP 툴킷 | 문장 단위 분할 → ICC 최고지만 BI 최저; 본 논문의 베이스라인 |
| **LumberChunker** (Duarte et al.) | 2024 | LLM으로 자연 경계 감지 | LLM 기반이나 단일 전략 강제; 본 논문의 LLM-Regex는 결정론적 정규식으로 효율화 |
| **AutoChunker** (Jain et al.) | 2025 | 구조화 텍스트 자동 청킹 + 평가 | 청킹 평가 체계 제안이라는 유사 동기; 단, 내재적 지표 5종 조합은 본 논문이 더 포괄적 |
| **Late Chunking** (Günther et al.) | 2025 | 긴 컨텍스트 임베딩으로 청크 임베딩 | 임베딩 수준 개선; 청킹 전략 선택은 다루지 않음 |
| **CrossFormer** (Ni et al.) | 2025 | 크로스 세그먼트 의미 융합 | 세그먼트 간 정보 전파; 청킹 자체보다 표현 학습 중심 |
| **MOC** (Zhao et al.) | 2025 | 혼합 청킹 학습자(Mixture of Chunking Learners) | 유사한 혼합 전략; 그러나 학습 기반 vs 본 논문의 지표 기반 |
| **Is Semantic Chunking Worth It?** (Qu et al.) | 2025 | 의미 청킹의 계산 비용 분석 | 의미 청킹이 비용 대비 효과가 제한적임을 지적; 본 논문도 동일한 관찰 후 대안 제시 |

### 4-2. RAG 평가 관련 연구

| 논문 | 연도 | 방법 | 비교 |
|------|------|------|------|
| **G-Eval** (Liu et al.) | 2023 | GPT-4 기반 NLG 평가 | 본 논문이 Answer Correctness 평가에 G-Eval 채택 |
| **DeepEval** (Confident AI) | 2025 | LLM 평가 프레임워크 | Retrieval Completeness, Answer Correctness 구현에 활용 |
| **Document Segmentation Matters for RAG** (Wang et al.) | 2025 | 문서 분할이 RAG에 미치는 영향 분석 | 분할의 중요성 인식 공유; 단, 구체적 적응 전략은 본 논문이 더 구체적 |
| **Reconstructing Context** (Merola & Singh) | 2025 | 고급 청킹 전략 비교 평가 | 다운스트림 평가 중심; 본 논문은 내재적 지표를 추가로 제안 |

### 4-3. 핵심 차별점 요약

```
기존 연구: 단일 청킹 전략 최적화 OR 다운스트림 평가만 사용
본 논문:   문서별 적응적 선택 + 내재적 지표 기반 직접 평가 + 새로운 청킹 기법
```

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5-1. 향후 연구에 미치는 영향

**① 청킹 평가 패러다임의 전환**

본 논문이 제안한 5가지 내재적 지표(RC, ICC, DCC, BI, SC)는 다운스트림 성능에 의존하지 않는 독립적 평가 체계를 확립함으로써, 향후 청킹 연구가 모델 선택이나 프롬프트 설계와 분리된 공정한 비교를 가능하게 한다. 이는 청킹 연구를 위한 **표준 벤치마크 수립**의 기초가 될 수 있다.

**② 문서 인식(Document-aware) RAG 설계 원칙 확립**

"적응적 선택이 단일 최선 방법보다 항상 우수하다"는 실험적 증거는, 향후 RAG 시스템 설계 시 **문서 특성을 1등급 시민(first-class citizen)**으로 취급해야 함을 강력히 시사한다.

**③ 후처리의 중요성 재인식**

포스트프로세싱만으로 SC를 최대 41%p 개선한 결과는, 기존의 복잡한 청킹 알고리즘보다 **간단한 정규화 후처리가 더 실용적**일 수 있음을 보여준다.

**④ 증폭 효과(Amplification Effect) 발견**

청킹 품질의 작은 개선($\Delta \approx 1\text{pp}$)이 RAG 성능에서 큰 차이($\Delta \approx 8\text{-}10\text{pp}$)로 이어진다는 발견은, 청킹이 RAG 최적화에서 **가장 높은 ROI를 가진 컴포넌트** 중 하나임을 시사한다.

---

### 5-2. 향후 연구 시 고려사항

**[방법론적 고려사항]**

1. **지표 가중치 최적화**
   현재 5개 지표는 동일 가중치로 평균됨. 도메인별/태스크별로 어떤 지표가 더 중요한지 학습하는 방향 필요:
   $$\text{Score}(d,c) = \sum_{i} w_i \cdot \text{metric}_i(d,c), \quad \sum_i w_i = 1$$
   where $w_i$는 학습 또는 도메인 전문가가 설정.

2. **쿼리 인식(Query-aware) 적응**
   현재 시스템은 인덱싱 시점에 고정 전략 선택. 쿼리 특성(사실형 vs 요약형 vs 추론형)에 따라 동적으로 청킹 방법을 조정하는 연구 필요.

3. **청크 선택 풀 확장**
   현재 4가지 청킹 방법만 후보로 사용. 계층적 청킹(Hierarchical), 중첩 청킹(Overlapping Window), 명제 기반 청킹(Propositional) 등을 후보 풀에 추가하면 더 다양한 문서 타입에 대한 적응 가능.

**[평가 체계 고려사항]**

4. **다국어 RC 지표 개발**
   Maverick의 영어 한정 제약을 극복하기 위해 mBERT 또는 XLM-R 기반 다국어 공참조 모델 연구 필요.

5. **대규모 벤치마크 구축**
   33개 문서는 통계적 안정성이 낮음. BEIR, MTEB 같은 표준 벤치마크와 연동한 대규모 청킹 평가 체계 구축이 필요.

6. **인간 평가 통합**
   G-Eval, Retrieval Completeness 모두 LLM-as-judge 방식. LLM 판단의 편향 및 일관성 문제를 보완하기 위한 인간 평가 비교 연구 필요.

**[시스템 효율성 고려사항]**

7. **계산 비용 최적화**
   - DCC: 슬라이딩 윈도우마다 임베딩 재계산 → 문서 수준에서 한 번 계산 후 매핑
   - RC: Maverick 배치 클러스터링 미지원 → 커스텀 배치 구현
   - 전처리 캐싱: 동일 문서에 대한 반복 임베딩 계산 회피

8. **사전 스크리닝(Pre-screening) 메커니즘**
   모든 후보 청킹 방법을 실행하는 대신, 문서 특성(길이, 구조 복잡도, 도메인)으로 후보를 사전 필터링하여 계산 비용 절감.

**[일반화 고려사항]**

9. **비공식 텍스트 및 멀티모달 문서**
   현재 법률·기술·사회과학 형식 문서에 한정. 대화체, 소셜미디어, 이미지-텍스트 혼합 문서에 대한 추가 검증 필요.

10. **대안 임베딩 모델과의 상호작용**
    현재 jina-embeddings-v3 고정 사용. 다양한 임베딩 모델에서 ICC/DCC가 일관된 결과를 보이는지 검증 필요.

---

## 참고자료 (출처)

**주 논문:**
- Paulo Roberto de Moura Júnior, Jean Lelong, Annabelle Blangero. *"Adaptive Chunking: Optimizing Chunking-Method Selection for RAG."* arXiv:2603.25333v1 [cs.CL], 26 Mar 2026. (논문 제출 날짜 기준)
- GitHub: https://github.com/ekimetrics/adaptive-chunking

**논문 내 인용 참고문헌:**
- Duarte et al. (2024). *LumberChunker: Long-form Narrative Document Segmentation.* EMNLP 2024 Findings.
- Günther et al. (2025). *Late Chunking: Contextual Chunk Embeddings Using Long-context Embedding Models.*
- Jain et al. (2025). *AutoChunker: Structured Text Chunking and Its Evaluation.* ACL 2025 Industry Track.
- Liu et al. (2023). *G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment.* EMNLP 2023.
- Martinelli et al. (2024). *Maverick: Efficient and Accurate Coreference Resolution.* ACL 2024.
- Merola & Singh (2025). *Reconstructing Context: Evaluating Advanced Chunking Strategies for RAG.*
- Ni et al. (2025). *CrossFormer: Cross-segment Semantic Fusion for Document Segmentation.* arXiv:2503.23671.
- Qi et al. (2020). *Stanza: A Python NLP Toolkit for Many Human Languages.* ACL 2020.
- Qu et al. (2025). *Is Semantic Chunking Worth the Computational Cost?* NAACL 2025 Findings.
- Smith & Troynikov (2024). *Evaluating Chunking Strategies for Retrieval.* Chroma Research.
- Sturua et al. (2024). *jina-embeddings-v3: Multilingual Embeddings with Task LoRA.*
- Wang et al. (2025). *Document Segmentation Matters for Retrieval-Augmented Generation.* ACL 2025 Findings.
- Zhao et al. (2025). *MOC: Mixtures of Text Chunking Learners for RAG.*
- Confident AI (2025a/b). DeepEval Framework Documentation.
- LangChain (2025a). RecursiveCharacterTextSplitter API Documentation.
- LangChain (2025b). SemanticChunker (Experimental) API Documentation.
- LlamaIndex (2024). Semantic Chunker Documentation.
- Microsoft (2025). Azure AI Document Intelligence.
