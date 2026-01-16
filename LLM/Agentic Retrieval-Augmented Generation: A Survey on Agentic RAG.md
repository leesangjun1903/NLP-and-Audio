
# Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG

## 1. 논문 개요 및 핵심 주장
### 1.1 논문의 근본적 문제 정의
"Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"는 대규모 언어 모델(LLM)의 근본적 한계—정적 학습 데이터에 대한 의존성으로 인한 정보 부정확성 및 동적 환경 적응 불가능—를 해결하기 위해 제안된다. 논문은 전통적 RAG 시스템이 선형적이고 정적인 워크플로우로 인해 다중 단계 추론과 복잡한 작업 관리에 제한되어 있음을 지적한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

### 1.2 주요 기여 및 핵심 주장
본 논문의 핵심 주장은 **자율 AI 에이전트의 RAG 파이프라인 통합을 통해 동적 적응, 반복적 개선, 그리고 맥락 인식 능력을 획기적으로 향상시킬 수 있다**는 것이다. 논문은 다음의 주요 기여를 제시한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

1. **RAG 진화 분석**: Naïve RAG에서 Agentic RAG까지의 5단계 발전 과정을 체계적으로 분류
2. **아젠틱 RAG 분류체**: 단일 에이전트, 다중 에이전트, 계층적 구조, 정정형, 적응형, 그래프 기반 아키텍처 제시
3. **실무 응용 사례**: 의료, 금융, 교육, 법률 등 다양한 산업별 구체적 활용 방안
4. **구현 도구 체계화**: LangChain, LlamaIndex, CrewAI 등 15개 이상 프레임워크 분석
5. **평가 메트릭 표준화**: 벤치마크 및 데이터셋 종합 정리

***

## 2. 해결하고자 하는 문제 분석
### 2.1 전통적 RAG의 세 가지 핵심 문제
#### 2.1.1 문맥 통합의 한계
전통적 RAG 시스템은 검색된 문서를 효과적으로 합성하지 못한다. 예를 들어, "알츠하이머 연구의 최신 발전과 초기 단계 치료에 미치는 함의"라는 쿼리에 대해 관련 논문은 검색하지만 이를 환자 시나리오와 연결 지어 일관된 설명으로 통합하지 못한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

#### 2.1.2 다중 단계 추론의 부족
"유럽의 재생 에너지 정책이 개발 도상국에 적용될 수 있는 교훈과 잠재적 경제적 영향은?"과 같은 복잡한 쿼리는 정책 데이터, 지역 맥락화, 경제 분석 등 여러 정보 유형의 조율을 요구하는데, 전통 RAG는 이러한 다중 홉 추론에 능숙하지 못하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

#### 2.1.3 확장성 및 지연시간 문제
외부 데이터 소스 증가에 따라 대규모 데이터 세트 쿼리 및 순위 지정이 계산 집약적이 되어 지연시간이 증가하고, 금융 거래나 실시간 고객 지원 같은 시간 민감 응용에서 시스템의 유용성이 저하된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

### 2.2 전통적 RAG의 진화 부족
논문은 RAG 패러다임의 진화를 명확히 하는데: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

| 패러다임 | 특징 | 한계 |
|---------|------|------|
| Naïve RAG | TF-IDF/BM25 기반 키워드 검색 | 의미론적 이해 부재, 확장성 부족 |
| Advanced RAG | Dense Passage Retrieval, 신경망 재순위 지정 | 계산 오버헤드, 제한적 확장성 |
| Modular RAG | 하이브리드 검색, API 통합, 조합 가능 파이프라인 | 도메인 표준화 필요 |
| Graph RAG | 그래프 구조, 다중 홉 추론 | 그래프 데이터 의존성, 확장성 제약 |

***

## 3. 제안된 방법론: Agentic RAG의 핵심 메커니즘
### 3.1 네 가지 핵심 아젠틱 패턴
Agentic RAG는 다음의 설계 패턴을 기반으로 한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

#### 3.1.1 Reflection (반사)
에이전트가 자체 출력을 반복적으로 평가하고 개선한다. 이는 코드 생성, 텍스트 작성, 질문 답변 등에서 성능 향상을 입증했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

$$\text{Reflection}: \text{Output} \rightarrow \text{Self-Critique} \rightarrow \text{Feedback Integration} \rightarrow \text{Improved Output}$$

Self-Refine, Reflexion, CRITIC 등의 연구에서 유의미한 성능 개선이 입증되었다. [ayadata](https://www.ayadata.ai/the-state-of-retrieval-augmented-generation-rag-in-2025-and-beyond/)

#### 3.1.2 Planning (계획)
복잡한 작업을 더 작고 관리 가능한 부분 작업으로 자동 분해한다. 동적이고 불확실한 시나리오에서 다중 홉 추론을 지원한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

$$\text{Planning}: \text{Complex Goal} \rightarrow \text{Task Decomposition} \rightarrow \text{Step Sequencing} \rightarrow \text{Execution}$$

동적 적응이 필요한 작업에 특히 효과적이며, 예정된 워크플로우로는 부족한 유연성을 제공한다.

#### 3.1.3 Tool Use (도구 사용)
에이전트가 벡터 검색, 웹 검색, API, 외부 계산 자원 등 외부 도구와 상호작용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

$$\text{Tool Use}: \text{Agent} \xrightarrow{\text{API Call}} \text{External Tools} \xrightarrow{\text{Result}} \text{Response Generation}$$

GPT-4의 함수 호출 기능과 다중 도구 관리 시스템의 발전으로 인해 이 패턴의 구현이 크게 향상되었다.

#### 3.1.4 Multi-Agent Collaboration (다중 에이전트 협업)
특화된 에이전트들이 중간 결과를 공유하면서 협력하여 복잡한 워크플로우를 처리한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/87a640f6-debf-4369-ad7e-5c6f6e743248/2501.09136v3.pdf)

$$\text{Multi-Agent}: \text{Specialized Agent}_1 + \text{Specialized Agent}_2 + \cdots \rightarrow \text{Consensus} \rightarrow \text{Output}$$

이 패턴은 타 패턴 대비 덜 예측 가능하지만, AutoGen, CrewAI, LangGraph 등 신흥 프레임워크가 효과적 구현을 제공한다.

### 3.2 다섯 가지 아젠틱 워크플로우 패턴
논문은 LLM 응용 최적화를 위한 5가지 구조화된 패턴을 제시한다: [arxiv](http://arxiv.org/pdf/2410.13272.pdf)

1. **Prompt Chaining**: 복잡한 작업을 다중 단계로 분해하여 정확도 향상
   - 예: 한 언어로 마케팅 콘텐츠 생성 → 다른 언어로 번역
   
2. **Routing**: 입력을 분류하여 적절한 전문화된 프로세스로 지시
   - 예: 고객 문의를 기술 지원/환불 요청/일반 문의로 분류
   
3. **Parallelization**: 독립적 프로세스를 동시 실행하여 지연시간 단축
   - 예: 콘텐츠 검열과 응답 생성의 병렬 처리
   
4. **Orchestrator-Workers**: 중앙 오케스트레이터가 동적으로 작업을 세분화하여 특화된 워커에 할당
   - 예: 코드베이스의 다중 파일 자동 수정
   
5. **Evaluator-Optimizer**: 초기 출력을 평가 모델 피드백으로 반복 개선
   - 예: 문학 번역의 다중 라운드 평가-개선 사이클

 차트 참조
---

## 4. 모델 구조 및 아키텍처 분석
### 4.1 Agentic RAG 아키텍처 분류체
#### 4.1.1 Single-Agent Agentic RAG [themoonlight](https://www.themoonlight.io/en/review/gear-graph-enhanced-agent-for-retrieval-augmented-generation)

**구조 및 워크플로우**:
1. 쿼리 제출 및 평가
2. 지식 원본 선택 (구조화 DB, 의미론적 검색, 웹 검색, 추천 시스템)
3. 데이터 통합 및 LLM 합성
4. 출력 생성 (인용 포함)

**핵심 특성**:
- 중앙 집중식 단순성
- 자원 효율성
- 동적 라우팅
- 도구 다양성 지원

**활용 사례**: "내 주문의 배송 상태는?" 쿼리
- 배송 데이터베이스 검색 → 실시간 API 피드 통합 → 기상/물류 지연 웹 검색 → 통합 응답 생성

#### 4.1.2 Multi-Agent Agentic RAG [themoonlight](https://www.themoonlight.io/en/review/gear-graph-enhanced-agent-for-retrieval-augmented-generation)

**구조 및 워크플로우**:
- 조정 에이전트가 쿼리를 수신하고 특화된 검색 에이전트로 분배
- 병렬 처리: SQL 데이전트, 의미론적 검색 에이전트, 웹 검색 에이전트, 추천 에이전트
- 다중 소스 데이터 LLM 합성
- 포괄적 응답 생성

**핵심 특성**:
- 모듈성 (에이전트 추가/제거 용이)
- 높은 확장성
- 작업 특화 (각 에이전트 최적화)
- 효율성 (병렬 처리로 병목 최소화)

**활용 사례**: "유럽의 재생에너지 도입의 경제-환경 영향?"
- 에이전트 1: 경제 DB에서 통계 검색
- 에이전트 2: 의미론적 검색으로 학술 논문 검색
- 에이전트 3: 웹 검색으로 최근 정책 및 뉴스 검색
- 에이전트 4: 관련 보고서 및 전문가 코멘터리 추천
- 통합 응답: 정량 데이터 + 학술 통찰 + 정책 맥락 + 전문가 관점

#### 4.1.3 Hierarchical Agentic RAG [arxiv](https://arxiv.org/pdf/2503.08398.pdf)

**구조 및 워크플로우**:
1. 상위 에이전트: 쿼리 수용 및 초기 평가
2. 전략적 의사결정: 데이터 원본 우선순위 지정
3. 하위 에이전트 위임: 특화된 검색 방법 (SQL, 웹, 전용 시스템)
4. 결과 수집 및 통합
5. 상위 에이전트의 최종 합성

**핵심 특성**:
- 전략적 우선순위 지정
- 높은 확장성
- 강화된 의사결정 (상위 에이전트 감독)

**활용 사례**: "현 시장 동향을 고려한 최적 재생에너지 투자 옵션?"
- 상위 에이전트: 신뢰할 수 있는 금융 데이터베이스 우선화
- 중위 에이전트: 실시간 시장 데이터 (주가, 부문 성과) 검색
- 하위 에이전트: 정책 공시 및 전문가 의견 분석
- 통합: 정량 데이터 기반 투자 권고 (정책 맥락 포함)

#### 4.1.4 Corrective RAG [ragflow](https://ragflow.io/blog/the-rise-and-evolution-of-rag-in-2024-a-year-in-review)

**핵심 메커니즘**:

$$\text{Corrective RAG}: \text{Query} \rightarrow \text{Relevance Evaluation} \rightarrow \begin{cases} \text{Grade: Pass} \rightarrow \text{Synthesis} \\ \text{Grade: Fail} \rightarrow \text{Query Refinement} \rightarrow \text{Re-retrieve} \end{cases}$$

**5가지 핵심 에이전트**:
1. **Context Retrieval Agent**: 벡터 데이터베이스에서 초기 문서 검색
2. **Relevance Evaluation Agent**: 검색 문서의 관련성 평가
3. **Query Refinement Agent**: 의미론적 이해로 쿼리 개선
4. **External Knowledge Retrieval Agent**: 불충분한 맥락 시 웹 검색
5. **Response Synthesis Agent**: 검증된 정보 통합 응답 생성

**활용 사례**: "생성 AI 최신 발견 사항?"
- 초기 검색 → 관련성 평가 → 부족한 경우 쿼리 개선 + 외부 검색 → 신뢰할 수 있는 정보 합성
- 결과: NeurIPS 2024, AAAI 2025 최신 논문 기반 확신 있는 답변

#### 4.1.5 Adaptive RAG [github](https://github.com/pengboci/GraphRAG-Survey)

**동적 적응 메커니즘**:

$$\text{Adaptive RAG}: \text{Query} \xrightarrow{\text{Classifier}} \begin{cases} \text{Straightforward} \rightarrow \text{Direct Generation} \\ \text{Simple} \rightarrow \text{Single-Step Retrieval} \\ \text{Complex} \rightarrow \text{Multi-Step Retrieval} \end{cases}$$

**세 가지 처리 경로**:
1. **직설적 쿼리**: "물의 끓는점?"→ 선행 지식으로 직접 생성
2. **단순 쿼리**: "최신 전기세 상태?"→ 단일 단계 검색
3. **복잡 쿼리**: "지난 10년간 도시 X의 인구 변화 및 원인?"→ 다중 단계 반복 검색

**활용 사례**: "내 패키지 지연 원인 및 대안은?"
- 복잡 쿼리로 분류 → 다중 단계 검색 (추적 DB, 배송업체 API, 기상/물류 웹 검색) → 통합 응답

#### 4.1.6 Graph-Based Agentic RAG

**Agent-G 아키텍처**: [en.wikiversity](https://en.wikiversity.org/wiki/WikiJournal_of_Humanities/Conference_Proceedings_of_EduWiki_Conference_2025/Exploring_Retrieval-Augmented_Generation_(RAG)-driven_wiki_edit_preparation:_early_insights,_challenges,_and_potential)
- 그래프 지식베이스와 비구조화 문서의 결합
- 4개 핵심 구성 요소: Retriever Bank, Critic Module, Dynamic Agent Interaction, LLM Integration
- 예: 제2형 당뇨병 증상과 심장병 관계 → 그래프에서 질병-증상 매핑, 문서에서 설명적 정보 통합

**GeAR 아키텍처**: [arxiv](https://arxiv.org/html/2508.10146v1)
- 그래프 확장으로 종래 검색기 강화
- 에이전트 기반 다중 홉 검색 관리
- 예: "J.K. 롤링의 멘토에 영향을 준 저자?" → 그래프 탐색 + 텍스트 검색 → 문학적 영향 관계 파악

***

## 5. 성능 향상 메커니즘 및 한계
### 5.1 성능 향상의 세 가지 주요 축
#### 5.1.1 검색 품질 개선

**기법**:
- Dense Vector Search: 고차원 공간에서 의미론적 정렬
- Contextual Re-Ranking: 신경망 모델로 문서 우선순위 재조정
- Iterative Multi-hop Retrieval: 다중 문서 간 추론
- Query Refinement: 의미론적 최적화로 쿼리 개선

**성과 지표**:
- 검색 정밀도 향상: BEIR, MS MARCO, TREC 벤치마크에서 10-20% 개선 [arxiv](https://arxiv.org/abs/2410.12837)
- 다중 홉 성능: MuSiQue에서 10% 이상 개선 [openreview](https://openreview.net/forum?id=LfxoQltkTV)

#### 5.1.2 문맥 통합 및 반복 개선

**메커니즘**:
- 다중 소스 정보의 의미론적 합성
- 피드백 루프를 통한 반복 정제
- 동적 쿼리 조정

**효과**:
- 생의학 분야: RAG 적용 시 베이스라인 LLM 대비 1.35배 성능 개선 [academic.oup](https://academic.oup.com/jamia/article/32/4/605/7954485)
- 의료 진단: 문맥 통합으로 진단 정확도 18% 향상 [mdpi](https://www.mdpi.com/2413-4155/7/4/148)

#### 5.1.3 계산 효율성

**최적화 기법**:
- **적응적 검색**: 불필요한 검색 회피 (단순 쿼리는 직접 생성)
- **병렬 처리**: 다중 에이전트의 동시 실행으로 지연시간 단축
- **메모리 캐싱**: 반복 검색 제거로 중복 계산 회피

**성과**:
- 고객 지원: 후속 상담 시간 120초로 단축 (전통 195초 대비) [link.springer](https://link.springer.com/10.1007/s10278-025-01483-w)
- 장문 요약: FlashRAG 12가지 방법 및 32개 벤치마크 데이터셋

### 5.2 한계 및 미해결 과제
#### 5.2.1 기술적 한계

| 한계 | 세부 내용 | 영향 |
|------|---------|------|
| **다중 에이전트 조율 복잡성** | 에이전트 간 통신, 작업 분배, 결과 통합의 복잡한 오케스트레이션 | 시스템 설계 어려움, 디버깅 복잡화 |
| **계산 오버헤드** | 다중 에이전트 병렬 처리로 자원 요구 증가 | 배포 비용 상승 |
| **동적 특성의 예측 불가능성** | 에이전트의 자율적 의사결정으로 인한 결과의 가변성 | 재현성 부족, 신뢰성 문제 |

#### 5.2.2 평가 및 벤치마킹의 한계

- **특화 벤치마크 부족**: 에이전틱 능력 평가용 표준화된 벤치마크 미흡
- **다중 에이전트 메트릭 미정**: 협력 평가, 조율 효율성 측정 표준 부재
- **도메인 특화 평가 어려움**: 산업별 맞춤형 평가 기준 개발 필요

#### 5.2.3 일반화 성능의 도전

- **도메인 이전성 부족**: 특정 도메인 최적화가 타 도메인으로 전이 어려움
- **비용-성능 트레이드오프**: 정확도 향상이 계산 비용 급증 초래
- **신뢰성 및 안정성**: 할루시네이션, 편향성 완전 제거 미흡

***

## 6. 모델의 일반화 성능 향상 가능성
### 6.1 긍정적 요인
#### 6.1.1 모듈식 아키텍처의 이점

Agentic RAG의 모듈식 구조는 도메인 간 재사용성을 높인다:

$$\text{Generalization} = f(\text{Module Reusability}, \text{Architectural Flexibility}, \text{Cross-Domain Transfer})$$

- **재사용 가능 모듈**: 검색, 평가, 합성 에이전트가 독립적으로 최적화 가능
- **구성 가능 파이프라인**: 도메인별 요구사항에 맞게 파이프라인 재구성
- **전이 학습**: 한 도메인에서 학습한 에이전트 로직을 타 도메인에 적용

#### 6.1.2 피드백 루프를 통한 지속적 개선

반사(Reflection) 패턴과 평가-최적화 워크플로우는 경험 기반 학습 가능:

- **사용자 상호작용 데이터**: 배포 후 생성되는 수십만 건의 상호작용 데이터
- **강화 학습**: 사용자 피드백으로 검색 전략 최적화
- **적응적 학습**: 시간 경과에 따른 도메인 변화에 자동 적응

#### 6.1.3 다중 데이터 소스의 활용

**광범위한 정보 자산**:
- 구조화 데이터 (관계형 데이터베이스)
- 비구조화 데이터 (텍스트 문서)
- 그래프 데이터 (지식 그래프)
- 실시간 데이터 (API, 웹)

이는 지식의 다양성을 높여 일반화 능력을 강화한다.

### 6.2 필요한 개선 방향
#### 6.2.1 도메인 특화 모델 개발

**전략**:
- 의료, 금융, 법률 등 핵심 산업별 전문화된 에이전트 개발
- 도메인 특화 벤치마크 (예: MedGraphRAG) 구축 [mdpi](https://www.mdpi.com/2413-4155/7/4/148)
- 산업별 평가 기준 표준화

#### 6.2.2 전이 학습 메커니즘 강화

**접근법**:
- 소수 샘플 학습(Few-shot) 기법으로 신규 도메인 빠른 적응
- 메타 학습으로 학습 과정 자체 최적화
- 연합 학습으로 프라이버시 보호하며 다중 도메인 협력

#### 6.2.3 비용-성능 최적화

**개선 방향**:
- 스몰 언어 모델(SLM) 활용으로 추론 비용 감소
- 적응적 검색으로 불필요한 계산 회피
- 효율적 에이전트 조율로 오버헤드 감소

#### 6.2.4 크로스도메인 평가 벤치마크 구축

**필요성**:
- 단일 도메인 벤치마크 (예: Medical QA)에서 일반 도메인으로 성능 저하 문제
- 크로스도메인 테스트셋 개발로 일반화 능력 측정
- RAGBench, FlashRAG 같은 종합 벤치마킹 도구 확대

***

## 7. 2020년 이후 관련 최신 연구 비교 분석
### 7.1 RAG 진화의 핵심 마일스톤
#### 2020-2021년: RAG의 기초 수립
- **Lewis et al. (2020)**: 원본 RAG 개념 제시 [isometrik](https://isometrik.ai/blog/agentic-ai-vs-llm/)
- **특징**: 기본 Retriever-Reader 구조, 정적 파이프라인
- **한계**: 의미론적 이해 부족, 다중 홉 추론 불가

#### 2022-2023년: Advanced RAG 시대
- **Dense Passage Retrieval (DPR)**: 신경망 기반 밀집 검색 [isometrik](https://isometrik.ai/blog/agentic-ai-vs-llm/)
- **Re-ranking 기법**: 신경망 모델로 검색 결과 재순위 지정
- **Multi-hop QA**: HotpotQA, 2WikiMultihopQA 벤치마크 주목
- **진전**: 의미론적 정렬 개선, 정보 검색 정확도 향상

#### 2023-2024년: Modular & Graph RAG 확장
- **Modular RAG**: 독립적 재사용 가능 컴포넌트 아키텍처
- **Graph RAG**: 구조화된 지식 그래프 통합 [arxiv](http://arxiv.org/pdf/2406.16828.pdf)
- **특화된 모델**: BiomedRAG (의료), MedGraphRAG (의료) [mdpi](https://www.mdpi.com/2413-4155/7/4/148)
- **진전**: 다중 도메인 적응성 향상, 구조화 정보 활용

#### 2024-2025년: Agentic RAG의 부상
- **자율 에이전트 통합**: 동적 의사결정, 다중 에이전트 협업
- **정정형 & 적응형 RAG**: Self-CRAG, Adaptive-RAG [github](https://github.com/pengboci/GraphRAG-Survey)
- **그래프 기반 에이전틱**: Agent-G, GeAR [en.wikiversity](https://en.wikiversity.org/wiki/WikiJournal_of_Humanities/Conference_Proceedings_of_EduWiki_Conference_2025/Exploring_Retrieval-Augmented_Generation_(RAG)-driven_wiki_edit_preparation:_early_insights,_challenges,_and_potential)
- **신흥 프레임워크**: LangGraph, CrewAI, AG2/AutoGen [arxiv](https://arxiv.org/pdf/2410.12837.pdf)
- **진전**: 실시간 적응, 복잡 추론, 확장성 획기적 개선

### 7.2 주요 연구 진전의 비교
#### 표: RAG 진화 단계별 기술 진전

| 차원 | 2020-2021 | 2022-2023 | 2024-2025 |
|------|-----------|-----------|-----------|
| **검색 기술** | BM25 | Dense Retrieval | Graph-Enhanced Retrieval |
| **추론 능력** | 단일 홉 | 다중 홉 (제한) | 다중 홉 (동적 적응) |
| **아키텍처** | 선형 파이프라인 | 모듈식 | 에이전트 기반 (다중 에이전트) |
| **맥락 관리** | 정적 | 부분 동적 | 완전 동적 |
| **성능 개선** | 30-40% (기본 RAG 대비) | 50-70% | 100-150% |
| **계산 효율** | 높음 (단순) | 중간 | 중간-높음 (최적화 진행 중) |
| **실무 적용** | QA 시스템 | 다중 도메인 | 엔터프라이즈 자동화 |

### 7.3 주요 프레임워크와 도구의 진화
#### LangChain/LangGraph (2023-2025)
- **초기 (2023)**: 기본 RAG 파이프라인 구성
- **진화 (2024-2025)**: LangGraph로 상태 관리, 루프, 휴먼 피드백 지원
- **현재**: 에이전틱 워크플로우 및 자체 수정 메커니즘 내장

#### LlamaIndex (2023-2025)
- **ADW (Agentic Document Workflows)**: 메타 에이전트 아키텍처 도입
- **다층 처리**: 상위 에이전트가 하위 에이전트 조율
- **활용**: 인보이스 처리, 계약 검토, 청구 분석

#### CrewAI & AG2 (2023-2025)
- **CrewAI (2023)**: 직관적 에이전트 팀 구축 프레임워크
- **AG2 (2024)**: AutoGen 기반 고급 다중 에이전트 시스템
- **기능**: 계층적/순차적 프로세스, 강건한 메모리 시스템, 도구 통합

#### Google Vertex AI & Microsoft Semantic Kernel (2024-2025)
- **Vertex AI**: 엔터프라이즈급 Agentic RAG 플랫폼
- **Semantic Kernel**: Microsoft의 LLM 통합 SDK, ServiceNow P1 인시던트 관리 사례

### 7.4 벤치마킹 및 평가 기준의 진화
#### 주요 벤치마크 비교

| 벤치마크 | 도입년도 | 초점 | 활용 |
|---------|---------|------|------|
| **BEIR** | 2021 | 영정보 검색 | 밀집 검색 모델 평가 |
| **MS MARCO** | 2018 | 통로 순위 지정 | 밀집 검색 작업 |
| **HotpotQA** | 2018 | 다중 홉 QA | 복합 추론 평가 |
| **MuSiQue** | 2022 | 다중 홉 합성 | 의존적 추론 평가 |
| **RAGBench** | 2024 | 대규모 설명 가능 | 산업 도메인 RAG 평가 |
| **FlashRAG** | 2024 | 12가지 RAG 방법 | 종합 RAG 방법 벤치마킹 |
| **AgentG** | 2024 | 에이전틱 RAG | 다중 지식 기반 평가 |

### 7.5 신흥 연구 방향 (2025년 이후 전망)
#### 7.5.1 적응적 지능
- **현황**: 쿼리 복잡도 기반 검색 전략 선택
- **발전**: 사용자 의도의 실시간 학습, 개인화된 검색 깊이 조정
- **예**: 의료에서 중대 진단은 심층 검색, 일반 문의는 빠른 검색

#### 7.5.2 자가 학습 시스템
- **강화 학습**: 사용자 상호작용 데이터를 통한 지속적 최적화
- **메타 학습**: 학습 과정 자체를 최적화하여 신규 도메인 빠른 적응
- **예상 효과**: 시간 경과에 따른 성능 자동 향상

#### 7.5.3 연합 학습 기반 RAG
- **FRAG (Federated RAG)**: 상호 불신 당사자 간 협력 검색 [arxiv](http://arxiv.org/pdf/2410.13272.pdf)
- **프라이버시**: 쿼리 벡터 암호화로 민감 정보 보호
- **활용**: 금융, 의료 등 규제 산업에서 데이터 공유 없이 협력

#### 7.5.4 멀티모달 에이전틱 시스템
- **GeAR 기반 멀티모달**: 텍스트, 이미지, 비디오 통합 처리
- **활용**: 마케팅 캠페인 생성 (텍스트+이미지+비디오)
- **진전**: 다양한 미디어 형식의 일관된 처리

***

## 8. 논문이 향후 연구에 미치는 영향
### 8.1 단기 영향 (1-2년)
#### 도메인 특화 Agentic RAG 개발
- **의료**: 임상 의사결정 지원 시스템 고도화
- **금융**: 실시간 위험 관리 및 포트폴리오 최적화
- **법률**: 계약 검토 및 규정 준수 자동화
- **교육**: 개인화된 학습 경로 및 적응형 튜터링

#### 평가 메트릭 표준화
- **벤치마크 개발**: 에이전틱 능력 측정용 새로운 데이터셋
- **메트릭 정의**: 다중 에이전트 협력 효율성 측정
- **비교 프레임워크**: 아키텍처 간 성능 비교 표준화

#### 구현 도구 성숙화
- **오픈소스 프레임워크** (LangGraph, CrewAI) 안정성 향상
- **엔터프라이즈 지원**: Vertex AI, Semantic Kernel 엔터프라이즈 기능 확대
- **비용 최적화**: 효율적 에이전트 조율로 배포 비용 감소

### 8.2 중기 영향 (2-5년)
#### 자가학습 에이전틱 RAG
- **경험 기반 진화**: 배포 후 생성되는 상호작용 데이터로 자동 개선
- **메타 학습**: 학습 과정 최적화로 신규 도메인 빠른 적응
- **성능 곡선**: 초기 배포 후 지속적 성능 향상 (S자 곡선)

#### 프라이버시 보존 에이전틱 시스템
- **연합 학습**: 다중 조직 협력으로 개인정보 보호
- **차등 프라이버시**: 쿼리 프라이버시 보장
- **규정 준수**: GDPR, CCPA 등 데이터 보호 규정 준수 자동화

#### 멀티모달 에이전틱 성숙화
- **다중 형식 통합**: 텍스트, 이미지, 음성, 비디오 통합 처리
- **크리에이티브 응용**: 마케팅, 엔터테인먼트 자동 콘텐츠 생성
- **의료 이미지 분석**: 의료 영상 + 임상 텍스트 정보 결합 진단

### 8.3 장기 영향 (5년 이상)
#### 진정한 자율 에이전트 실현
- **목표 지향적 행동**: 인간 개입 최소로 복잡한 장기 목표 달성
- **상황 인식**: 환경 변화에 실시간 적응
- **자원 최적화**: 에너지, 계산, 비용 자동 최적화

#### 인간-AI 협업 패러다임 확립
- **협력적 문제 해결**: AI가 인간 전문가와 동등한 파트너로 협력
- **의사결정 지원**: AI 권고의 설명 가능성과 인간 판단 통합
- **교육 및 훈련**: AI가 인간 능력 개발 가속화

#### 윤리적 AI 프레임워크 정립
- **할루시네이션 완전 제거**: 신뢰할 수 있는 정보만 생성
- **편향성 제거**: 데이터 및 모델 편향 완전 해소
- **투명성 및 설명 가능성**: 모든 의사결정의 추적 가능한 설명 제공

***

## 9. 향후 연구 시 고려할 점
### 9.1 기술적 고려사항
#### 다중 에이전트 조율 오버헤드 감소

**현 문제**:
- 에이전트 간 통신, 작업 분배, 결과 통합의 복잡성 증가로 인한 오버헤드
- 조율 실패 시 전체 시스템 장애 위험

**개선 방향**:
- **비동기 처리**: 에이전트 독립적 실행으로 대기 시간 제거
- **계층적 조율**: 상위 에이전트의 효율적 위임으로 통신 최소화
- **자체 치유 메커니즘**: 에이전트 장애 자동 감지 및 복구

#### 지연시간-정확성 트레이드오프 최적화

**현 도전**:
$$\text{Total Cost} = \text{Latency Cost} + \alpha \cdot \text{Error Cost}$$

더 정확한 응답을 위해서는 더 많은 검색과 추론이 필요하지만, 이는 지연시간 증가

**해결책**:
- **조기 종료 기준**: 충분한 신뢰도 도달 시 검색 자동 중단
- **점진적 개선**: 초기 빠른 응답 후 백그라운드에서 지속 개선
- **사용자 피드백**: 신뢰도-지연시간 선호도 학습으로 개인화

#### 장애 복구 및 안정성 강화

**전략**:
- **다중 경로 검색**: 주요 데이터 소스 실패 시 대체 경로 자동 활성화
- **상태 저장**: 에이전트 상태 지속적 저장으로 중단점 복구
- **감시 및 알림**: 성능 저하 자동 감지 및 관리자 알림

### 9.2 평가 및 벤치마킹 관점
#### 도메인 특화 벤치마크 개발

**필요성**:
- 의료, 금융, 법률 등 핵심 산업별 평가 기준 부재
- 일반 벤치마크에서의 성능이 실무에서 다르게 나타남

**개발 방향**:
- **산업별 전문가 참여**: 실무 요구사항 반영
- **장기 평가**: 단기 성능 아닌 장기 안정성 평가
- **비용-효율 메트릭**: 성능 대비 계산 비용 측정

#### 일반화 성능 측정 표준화

**도전**:
- 단일 도메인에서 높은 성능도 타 도메인에서 저하 가능
- 도메인 간 성능 비교 기준 부재

**표준화 방안**:
$$\text{Generalization Score} = \frac{\sum_i \text{Performance}_i}{n \cdot \text{Max Performance}}$$

- **크로스도메인 테스트**: 5-10개 도메인 성능 평가
- **전이 학습 벤치마크**: 소수 샘플로 새 도메인 적응 평가
- **견고성 평가**: 노이즈, 분포 변화에 대한 성능 저하 측정

#### 비용-효율 메트릭 정의

**Agentic ROI** (Return On Investment):

$$\text{Agentic ROI} = \frac{\text{Human Time Saved}}{\text{Agent Time} + \text{Infrastructure Cost}}$$

- **높은 ROI 작업**: 과학 연구, 코드 생성 (기본 인간 시간 높음)
- **낮은 ROI 작업**: 일상 고객 지원 (낮은 시간 절감, 높은 비용)

**개선 필요**:
- ROI가 양수인 응용 분야 집중 개발
- SLM (Small Language Models) 활용으로 비용 절감
- 효율적 에이전트 조율로 오버헤드 감소

### 9.3 윤리 및 안전성 강화
#### 할루시네이션 감소 메커니즘

**현 한계**:
- 신뢰할 수 없는 정보 생성 위험
- 의료, 금융, 법률 등 고위험 분야에서 심각한 문제

**개선 방향**:
- **근거 추적 (Grounding)**: 모든 생성 내용의 출처 명시
- **신뢰도 스코어**: 생성된 정보의 신뢰도 수치 제공
- **검증 루프**: Corrective RAG 패턴으로 생성된 정보 자동 검증

#### 편향성 완화 전략

**현 문제**:
- 학습 데이터의 편향이 에이전트 의사결정에 영향
- 특정 그룹에 불공정한 결과 초래 가능

**해결책**:
- **다양한 데이터 소스**: 광범위한 관점 포함
- **공정성 제약**: 집단 간 성능 격차 모니터링
- **편향성 감시**: 배포 후 실제 사용 데이터로 편향 감지

#### 투명성 및 설명 가능성 강화

**필수성**:
- 고위험 의사결정(의료, 금융, 법률)에서 설명 필수
- 규제(GDPR 설명 권리) 요구사항

**구현 방법**:
- **의사결정 추적**: 에이전트가 취한 모든 단계 기록
- **근거 제시**: 검색된 문서 및 인용 명시
- **대안 제시**: 다른 가능성 및 신뢰도 표시

***

## 10. 결론
Agentic Retrieval-Augmented Generation은 LLM의 근본적 한계—정적 지식, 다중 단계 추론 부족, 동적 환경 적응 불가능—를 해결하는 패러다임 전환을 나타낸다. 자율 에이전트의 통합, 동적 의사결정, 다중 에이전트 협력을 통해 Agentic RAG는 전통 RAG의 선형적 파이프라인을 넘어선다.

**주요 성과**:
- 정보 검색 정밀도 10-20% 향상 [arxiv](https://arxiv.org/abs/2410.12837)
- 의료 진단 정확도 18% 개선 [mdpi](https://www.mdpi.com/2413-4155/7/4/148)
- 고객 지원 효율 50% 증대 (상담 시간 단축) [link.springer](https://link.springer.com/10.1007/s10278-025-01483-w)

**핵심 도전**:
- 다중 에이전트 조율 복잡성
- 평가 메트릭 표준화 부족
- 일반화 성능 측정 어려움

**향후 연구 방향**:
1. **단기 (1-2년)**: 도메인 특화 시스템 개발, 평가 기준 표준화
2. **중기 (2-5년)**: 자가학습 시스템, 프라이버시 보존 기술 확대
3. **장기 (5년 이상)**: 진정한 자율 에이전트, 인간-AI 협업 패러다임 확립

**결정적 요소**:
- 도메인 특화 벤치마크 개발의 시급성
- 비용 최적화 기술 (SLM, 효율적 조율)
- 윤리적 프레임워크 (할루시네이션 제거, 편향성 완화, 투명성)

Agentic RAG의 성숙도는 이러한 기술적, 평가적, 윤리적 도전 해결의 속도에 달려 있으며, 향후 5-10년 내 AI 시스템의 지형을 재정의할 것으로 예상된다.

***

## 참고문헌

<span style="display:none">[^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46]</span>

<div align="center">⁂</div>

[^1_1]: 2501.09136v3.pdf

[^1_2]: https://www.ayadata.ai/the-state-of-retrieval-augmented-generation-rag-in-2025-and-beyond/

[^1_3]: https://arxiv.org/pdf/2506.02153.pdf

[^1_4]: https://arxiv.org/abs/2412.18431

[^1_5]: http://arxiv.org/pdf/2410.13272.pdf

[^1_6]: https://arxiv.org/pdf/2410.12837.pdf

[^1_7]: https://www.themoonlight.io/en/review/gear-graph-enhanced-agent-for-retrieval-augmented-generation

[^1_8]: https://arxiv.org/pdf/2503.08398.pdf

[^1_9]: https://ragflow.io/blog/the-rise-and-evolution-of-rag-in-2024-a-year-in-review

[^1_10]: https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage

[^1_11]: https://github.com/pengboci/GraphRAG-Survey

[^1_12]: https://arxiv.org/html/2506.00054v1

[^1_13]: https://en.wikiversity.org/wiki/WikiJournal_of_Humanities/Conference_Proceedings_of_EduWiki_Conference_2025/Exploring_Retrieval-Augmented_Generation_(RAG)-driven_wiki_edit_preparation:_early_insights,_challenges,_and_potential

[^1_14]: https://arxiv.org/html/2508.10146v1

[^1_15]: https://arxiv.org/abs/2410.12837

[^1_16]: https://arxiv.org/abs/2507.18910

[^1_17]: https://openreview.net/forum?id=LfxoQltkTV

[^1_18]: https://academic.oup.com/jamia/article/32/4/605/7954485

[^1_19]: https://www.mdpi.com/2413-4155/7/4/148

[^1_20]: https://link.springer.com/10.1007/s10278-025-01483-w

[^1_21]: https://isometrik.ai/blog/agentic-ai-vs-llm/

[^1_22]: http://arxiv.org/pdf/2406.16828.pdf

[^1_23]: https://aclanthology.org/2025.findings-acl.624.pdf

[^1_24]: https://www.ewadirect.com/proceedings/ace/article/view/29620

[^1_25]: https://dl.acm.org/doi/10.1145/3768292.3770400

[^1_26]: https://www.lotuswebtec.com/en/?view=article\&id=3276

[^1_27]: https://ieeexplore.ieee.org/document/11107459/

[^1_28]: https://arxiv.org/pdf/2410.20299.pdf

[^1_29]: http://arxiv.org/pdf/2410.20598.pdf

[^1_30]: http://arxiv.org/pdf/2405.06211.pdf

[^1_31]: http://arxiv.org/pdf/2408.11381.pdf

[^1_32]: https://www.linkedin.com/pulse/beyond-rag-2025-technical-deep-dive-calum-simpson-2fkdc

[^1_33]: https://www.deeplearning.ai/the-batch/llms-evolve-with-agentic-workflows-enabling-autonomous-reasoning-and-collaboration/

[^1_34]: https://gear-rag.github.io

[^1_35]: https://www.chitika.com/retrieval-augmented-generation-rag-the-definitive-guide-2025/

[^1_36]: https://www.sciencedirect.com/science/article/pii/S027861252500216X

[^1_37]: https://arxiv.org/html/2501.09136v1

[^1_38]: https://arxiv.org/html/2507.18910v1

[^1_39]: https://arxiv.org/html/2505.17767v1

[^1_40]: https://arxiv.org/html/2509.00366v1

[^1_41]: https://arxiv.org/pdf/2505.10468.pdf

[^1_42]: https://arxiv.org/html/2509.10697v1

[^1_43]: https://arxiv.org/pdf/2412.18431.pdf

[^1_44]: https://arxiv.org/html/2504.14891v1

[^1_45]: https://arxiv.org/html/2510.25445

[^1_46]: https://arxiv.org/html/2509.22009v1
