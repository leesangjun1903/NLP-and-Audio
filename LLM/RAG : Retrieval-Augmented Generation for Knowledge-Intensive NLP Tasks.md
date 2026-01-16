
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
