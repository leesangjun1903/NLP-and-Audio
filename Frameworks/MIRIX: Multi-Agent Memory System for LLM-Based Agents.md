# MIRIX: Multi-Agent Memory System for LLM-Based Agents

### 1. 핵심 주장과 주요 기여

MIRIX는 LLM 기반 에이전트의 가장 근본적인 문제를 해결하고자 한다: **언어 모델이 진정한 의미에서 기억하는 능력**. 기존의 메모리 시스템들이 평탄한(flat) 구조에 의존하여 사용자별 정보의 개인화, 추상화, 신뢰할 수 있는 회상을 제한하고 있다는 점에서 출발한다.[1]

**주요 기여**는 다음과 같다:[1]

- **6개 메모리 컴포넌트 기반 혁신적 아키텍처** 제안: 기존의 단순 short-term/long-term 메모리 이분법을 벗어나 Core Memory, Episodic Memory, Semantic Memory, Procedural Memory, Resource Memory, Knowledge Vault로 구성된 특화된 메모리 시스템 도입
- **멀티모달 지원**: 텍스트 중심의 기존 방식을 넘어 시각적, 멀티모달 경험을 포괄
- **저장 효율성**: 99.9%의 저장 공간 감소 달성 (ScreenshotVQA 벤치마크)
- **신규 벤치마크 제시**: 20,000개의 고해상도 스크린샷으로 구성된 ScreenshotVQA 데이터셋 소개
- **SOTA 성능**: LOCOMO 벤치마크에서 85.4%의 state-of-the-art 정확도 달성

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 기존 메모리 시스템의 문제점

기존 시스템들(Zep, Cognee, Letta, Mem0, ChatGPT)이 직면한 세 가지 핵심 문제:[1]

1. **구성적 메모리 구조 부재**: 모든 역사적 데이터를 단일 평탄 저장소에 보관하여 특화된 메모리 유형(절차적, 삽화적, 의미적)으로의 라우팅이 불가능하고, 회상 비효율성 야기
2. **멀티모달 지원 부족**: 텍스트 중심 메커니즘이 이미지, 인터페이스 레이아웃, 지도 등 비언어적 입력에 취약
3. **확장성 및 추상화 부족**: 특히 이미지와 같은 원본 입력 저장으로 인한 금지된 메모리 요구사항, 그리고 중요한 정보만을 요약하는 효과적인 추상화 계층 부재

#### 2.2 MIRIX의 6개 메모리 컴포넌트 구조[1]

| 메모리 유형 | 기능 | 데이터 구조 |
|-----------|------|-----------|
| **Core Memory** | 항상 가시적이어야 하는 고우선순위 정보 (에이전트 persona, 사용자 속성) | persona 블록, human 블록 |
| **Episodic Memory** | 시간-소인된 사건 및 일시적으로 근거한 사용자 상호작용 | event_type, summary, details, actor, timestamp |
| **Semantic Memory** | 시간/사건과 무관한 추상적 지식 및 사실 정보 | name, summary, details, source |
| **Procedural Memory** | 목표-지향적 프로세스(how-to 가이드, 작업흐름) | entry_type, description, steps |
| **Resource Memory** | 사용자가 활발히 사용하는 문서, 파일, 멀티미디어 | title, summary, resource_type, content |
| **Knowledge Vault** | 자격증명, 주소, API 키 등 정확히 보존해야 하는 민감 정보 | entry_type, source, sensitivity_level, secret_value |

#### 2.3 Active Retrieval 메커니즘[1]

기존 시스템의 한계를 극복하기 위해 MIRIX는 **Active Retrieval** 메커니즘을 제안한다:

$$\text{Topic} = \text{LLM}(\text{Context}, \text{Query})$$

$$\text{Retrieved Memory} = \bigcup_{i=1}^{6} \text{Retrieve}_i(\text{Topic})$$

$$\text{Response} = \text{LLM}(\text{Query} + \text{Retrieved Memory})$$

이 접근법은 두 가지 단계로 작동한다:[1]

1. **주제 생성**: 에이전트가 입력 문맥에 기반하여 현재 주제를 생성
2. **메모리 회수**: 각 메모리 컴포넌트에서 관련 정보를 검색하고, 6개 메모리 타입 모두에서 top-10 관련 항목 회수

예를 들어, "Twitter의 CEO는 누구인가?"라는 쿼리에 대해, 에이전트는 "CEO of Twitter"라는 주제를 추론하고 이를 사용하여 모든 메모리 컴포넌트에서 관련 항목들을 검색한다.[1]

#### 2.4 다양한 회수 함수[1]

MIRIX는 단일 회수 방식을 넘어 상황에 적응적인 여러 회수 함수를 지원한다:

- **embedding_match**: 의미적 유사성 기반 (벡터 데이터베이스)
- **bm25_match**: BM25 유사도 기반 키워드 매칭
- **string_match**: 정확한 문자열 매칭

***

### 3. 모델 구조: 멀티에이전트 워크플로우[1]

#### 3.1 아키텍처 개요

MIRIX는 8개의 특화된 에이전트로 구성된다:

$$\text{MIRIX} = \{\text{Meta Memory Manager}\} \cup \bigcup_{i=1}^{6} \{\text{Memory Manager}_i\} \cup \{\text{Chat Agent}\}$$

각 Memory Manager는 담당 메모리 컴포넌트의 업데이트 및 회수를 관리하고, Meta Memory Manager는 전체 라우팅과 조율을 담당한다.[1]

#### 3.2 메모리 업데이트 워크플로우[1]

```
사용자 입력 → [메모리베이스에 자동 검색] → Meta Memory Manager
                                    ↓
                [라우팅 분석] → 관련 Memory Managers (병렬 처리)
                                    ↓
                        [개별 메모리 업데이트]
                                    ↓
                        [중복 제거 검증]
                                    ↓
                    Meta Memory Manager ← 완료 보고
```

Meta Memory Manager는 새로운 입력을 분석하여 6개의 메모리 컴포넌트 중 어느 것이 관련이 있는지 판단하고, 해당 Memory Managers로 라우팅한다. 이 메모리 업데이트 과정은 보통 20개의 고유한 스크린샷이 수집된 후 트리거되는데, 이는 약 60초마다 발생한다.[1]

#### 3.3 대화형 회수 워크플로우[1]

```
사용자 쿼리 → Chat Agent
              ↓
          [초기 메모리 검색: 모든 6개 메모리에서 고수준 요약 회수]
              ↓
          [쿼리 분석: 더 정교한 검색이 필요한 메모리 컴포넌트 판단]
              ↓
          [타겟 검색: 적절한 회수 함수 선택 및 실행]
              ↓
          [정보 통합 및 응답 생성]
```

Chat Agent는 사용자의 메모리 업데이트 요청도 직접 처리할 수 있으며, 특정 메모리 컴포넌트의 정확한 수정을 가능하게 한다.[1]

***

### 4. 성능 향상 분석

#### 4.1 ScreenshotVQA 벤치마크 결과

MIRIX는 새로운 멀티모달 벤치마크를 구성했으며, 다음의 특징이 있다:[1]

- **데이터셋 규모**: 3명의 박사 학생 활동 기록, 총 5,886~18,178개의 고해상도 스크린샷 (2K-4K 해상도)
- **평가 질문**: 87개 수동 작성 질문

성능 결과:[1]

| 방법 | 정확도 ↑ | 저장 요구량 ↓ |
|-----|---------|------------|
| Gemini (장문맥) | 11.66% | 236.70MB |
| SigLIP@50 (RAG) | 44.10% | 15.07GB |
| **MIRIX** | **59.50%** | **15.89MB** |

**개선율**:
- RAG 대비 **35% 정확도 향상**, **99.9% 저장 감소**
- 장문맥 모델 대비 **410% 정확도 향상**, **93.3% 저장 감소**[1]

#### 4.2 LOCOMO 벤치마크 결과

LOCOMO는 10개의 긴 대화, 대화당 600개 대화 턴, 약 26,000개 토큰으로 구성되며, 추출된 메모리만으로 200개 질문에 답하도록 제한된 평가 설정이다.[1]

질문 유형별 성능:[1]

| 질문 유형 | 단일 홉 | 다중 홉 | 개방형 | 시간적 | 전체 |
|---------|--------|--------|-------|--------|------|
| LangMem | 74.47 | 61.06 | 67.71 | 86.92 | 78.05 |
| Zep | 79.43 | 69.16 | 73.96 | 83.33 | 79.09 |
| Mem0 | 62.41 | 57.32 | 44.79 | 66.47 | 62.47 |
| **MIRIX** | **85.11** | **83.70** | **65.62** | **88.39** | **85.38** |
| Full-Context | 88.53 | 77.70 | 71.88 | 92.70 | 87.52 |

MIRIX는 **전체 정확도에서 기존 최강 경쟁자(LangMem)보다 8점 이상 향상**.[1]

#### 4.3 질문 유형별 성능 분석[1]

- **단일 홉 & 시간적 질문**: MIRIX는 계층적 메모리 저장의 효과성을 검증하면서 Full-Context와의 작은 격차는 질문의 모호성(예: 예정된 캠핑 날짜 vs 실제 캠핑 사건)에서 기인
- **다중 홉 질문**: **가장 큰 성능 향상** (24점 이상). 예: "4년 전 Caroline이 이사 온 곳은?" → 답: Sweden. MIRIX는 "Caroline moved from her hometown, Sweden, 4 years ago"로 통합된 사건을 저장하여 실시간에 여러 정보를 연결할 필요가 없음
- **개방형 질문**: Full-Context와의 격차 (약 6점) 존재. RAG 기반 메서드의 내재적 한계인 "글로벌 이해 부족"에서 기인[1]

***

### 5. 일반화 성능 향상 가능성

#### 5.1 크로스 도메인 적용 가능성

MIRIX의 6개 컴포넌트 설계는 **도메인별 요구사항에 자연스럽게 적응**할 수 있는 구조다:[1]

- **의료 도메인**: Procedural Memory로 의료 절차 저장, Semantic Memory로 질병/약물 관계 저장, Episodic Memory로 환자 병력 추적
- **금융 도메인**: Knowledge Vault로 민감한 계정 정보, Semantic Memory로 시장 개념, Resource Memory로 재정 문서 저장
- **웨어러블 디바이스**: 계산 및 저장 제약을 고려한 **하이브리드 온디바이스/클라우드 메모리 관리**[1]

#### 5.2 더 많은 데이터에의 스케일링

MIRIX는 구조화된 메모리 접근으로 인해 **확장성 우위**를 가진다:

- 저장 효율성: 원본 이미지의 99.9% 저장 감소
- 회수 효율성: 6개 컴포넌트별 특화된 회수로 불필요한 정보 검색 최소화
- 병렬 처리: 6개 Memory Managers의 병렬 업데이트로 처리 시간 최적화[1]

#### 5.3 새로운 사용자로의 전이 학습(Transfer Learning)

MIRIX는 **메모리 마이그레이션 메커니즘**을 통해 사용자 간 지식 공유가 가능하다:[1]

- Semantic Memory와 Procedural Memory는 여러 사용자에게 공유 가능
- Core Memory만 사용자별로 개인화
- 이는 새로운 사용자 온보딩 시 학습 곡선을 단축

#### 5.4 개방형 질문에서의 개선 가능성

현재 MIRIX가 개방형 질문에서 Full-Context보다 약 6점 낮은 성능을 보이는 것은, 다음의 개선으로 해결 가능하다:[1]

- **의미적 통합 메모리**: 현재 6개 컴포넌트를 넘어 "관계 네트워크" 컴포넌트 추가로 다중 사건 간의 인과 관계 표현
- **강화 학습**: Reinforcement Learning으로 어떤 메모리 유형이 어떤 질문에 가장 효과적인지 학습[2]
- **추론 기반 회수**: 현재 simple RAG 기반 회수 대신 reasoning 모델을 활용한 복합 추론[3]

***

### 6. 현재 한계점

#### 6.1 실험상 한계[1]

1. **ScreenshotVQA 데이터셋 규모**: 3명 사용자, 87개 질문으로 일반화 검증 제한적
2. **모델 의존성**: Gemini-2.5-flash (ScreenshotVQA), GPT-4.1-mini (LOCOMO)에 성능 의존
3. **오픈 도메인 성능**: Full-Context와 약 6점 격차 존재
4. **비용-효율성 미분석**: 6개 에이전트 간의 통신 오버헤드 및 API 호출 비용 정량화 부재

#### 6.2 기술상 한계[1]

1. **구조적 재구성의 어려움**: 메모리 컴포넌트의 경계가 명확하지 않은 정보의 처리 (예: 감정 상태는 Episodic인가, Core인가?)
2. **메모리 충돌 해결**: 서로 다른 Memory Managers가 동시에 같은 정보 업데이트 시도 시 충돌 해결 메커니즘 미흡
3. **장기 메모리 응축**: Core Memory가 90% 용량 도달 시 "통제된 재작성" 과정이 중요 정보 손실 가능성

#### 6.3 일반화 관련 한계[1]

1. **모달리티 편향**: 시각적 입력(ScreenshotVQA)과 텍스트 입력(LOCOMO)에만 평가됨, 음성/비디오 등 다른 모달리티 검증 부재
2. **도메인 적응성**: 의료, 금융 등 특화 도메인에서의 성능 검증 없음
3. **개방형 추론**: Long-range reasoning 요구하는 복잡한 다중 홉 추론에서 여전히 Full-Context보다 6% 낮음

***

### 7. 향후 연구 영향 및 고려사항

#### 7.1 단기 연구 방향

**멀티모달 메모리 확장**[4][3]
기존 텍스트-시각 통합에서 나아가, 음성(오디오) 및 비디오 입력의 네이티브 지원이 필수다. 특히 웨어러블 디바이스 시장의 급성장을 고려하면, Real-time audio processing과 video summarization이 핵심이다.[3][1]

**메모리 보안 및 프라이버시**[5][1]
Knowledge Vault의 민감 정보 보호를 위한 암호화 및 접근 제어 강화가 필요하다. MIRIX가 제시한 3층 프라이버시 아키텍처(암호화 계층, 프라이버시 제어, 분산 저장소)는 진일보한 실행이 필요하다. 특히 엔터프라이즈 환경에서의 HIPAA/GDPR 준수 검증이 중요하다.[5][1]

**메모리 베이스라인 개선**[6][7]
현재 MIRIX는 LLM-as-a-Judge 평가만 사용하는데, 더 객관적인 평가 지표(정밀도, 재현율, F1 점수)와 **MemAE (Memory Agent Evaluation) 같은 종합 평가 프레임워크** 도입이 필요하다. MemAE는 회수 정확도, 테스트-타임 학습, 장거리 이해, 충돌 해결 4가지 핵심 역량을 평가한다.[7]

#### 7.2 중기 연구 방향

**신경망 기반 메모리 관리자**[8][2]
현재 MIRIX의 Meta Memory Manager는 LLM 기반이나, 이를 **학습 가능한 신경 네트워크**로 대체하여 라우팅 효율성 향상 가능. 특히 **Reinforcement Learning (RL)**을 활용한 동적 라우팅 최적화가 주목된다.[2]

**장기 메모리의 지속적 응축**[9]
현재 핵심 메모리의 90% 임계값 기반 재작성은 정보 손실 위험. **재귀적 요약 (Recursive Summarization)**을 통해 보다 우아한 메모리 압축이 가능하다. 예: 여러 에피소드를 계층적으로 요약하여 상위 메모리 집계본 생성.[9]

**다중 에이전트 메모리 협력**[10][5]
MIRIX의 6개 Memory Manager는 현재 독립적으로 작동하지만, **메모리 공유(Memory Sharing)** 프레임워크를 도입하면:[5]
- 에이전트 간 정보 동기화 개선
- 시간 민감 정보의 우선순위 조정
- 다중 에이전트 시스템의 확장 가능성[10]

**개방형 추론 향상**[11]
현재 개방형 질문에서 Full-Context 대비 6점 격차는, **LLM 추론 모델 (Reasoning Models)** 도입으로 극복 가능. 예: OpenAI-O3, DeepSeek-R1 같은 추론 특화 모델의 적용.[11]

#### 7.3 장기 연구 방향

**MIRIX 마켓플레이스 생태계**[1]
MIRIX가 제시한 "Agent Memory Marketplace" 비전은 혁신적이다:[1]
- 개인 메모리의 디지털 자산화
- 프라이버시-보존 메모리 거래
- 전문가 커뮤니티 기반 집단 지능 구축
이는 다음 세대 AI 경제에서 **인간 경험 데이터의 가치 재정의**를 의미한다.[1]

**신경-상징 메모리 통합**[12]
Long-term 파라미터 메모리(신경망의 가중치)와 episodic 회수(벡터 DB 기반)의 통합. 예: **M+ Framework**는 scalable long-term memory와 함께 parametric knowledge를 결합하여 일반화 능력 향상.

**웨어러블 & 엣지 디바이스 최적화**[13][1]
MIRIX의 하이브리드 온디바이스/클라우드 아키텍처는 **AI 안경, AI 핀, 스마트워치** 같은 차세대 웨어러블에 필수. 특히:[1]
- 저전력 메모리 업데이트
- 실시간 프라이버시 보호
- 엣지-클라우드 동기화[13]

**지속적 학습과 메모리 진화**
MIRIX는 현재 정적 메모리 구조이나, 새로운 메모리 유형의 **자동 생성/제거**와 사용자 행동 진화에 따른 동적 조정이 가능해야 한다. **Procedural Memory는 반복 사용으로 Semantic으로 승격되거나**, 사용되지 않는 Episodic 메모리는 자동 응축.

#### 7.4 최신 연구 트렌드와의 연계

**Multi-Agent 생태계 통합**[14][4][3][10][5]
최근 연구는 MIRIX 같은 구조화된 메모리 시스템이 **협력적 멀티에이전트 시스템**의 필수 기반임을 입증한다. AgentSquare는 모듈식 설계 공간에서 Planning, Reasoning, Tool Use와 함께 **Memory를 4대 핵심 모듈**로 명시한다.[4][14][3]

**메모리 공학(Memory Engineering)**[5]
MongoDB의 최신 연구는 멀티에이전트 시스템의 실패가 **통신 문제가 아닌 메모리 문제**임을 지적한다. MIRIX의 Active Retrieval과 메모리 라우팅은 이 "메모리 위기"의 직접적 해결안이다.[5]

**Reasoning 모델과의 결합**[3][2]
DeepSeek-R1, OpenAI-O3 같은 reasoning 모델의 등장으로, MIRIX의 메모리 시스템이 더욱 강화될 수 있다. 예: Reasoning 모델이 메모리 검색 전략을 개선하거나, Complex Multi-hop 질문에 대한 추론 경로를 최적화.[2][3]

***

### 8. 결론

MIRIX는 LLM 에이전트의 메모리 문제를 **구조화된 멀티컴포넌트 아키텍처**로 혁신적으로 해결한다. 6개의 특화된 메모리 타입과 멀티에이전트 워크플로우는:

- **효율성**: 99.9% 저장 감소
- **성능**: LOCOMO에서 85.38% SOTA 정확도
- **확장성**: 도메인 및 모달리티 확장 가능성
- **실용성**: 웨어러블 디바이스부터 엔터프라이즈까지 적용 범위

향후 연구는 **멀티모달 통합, 신경 기반 메모리 관리, Reasoning 모델 결합, 지속적 메모리 진화** 등을 중심으로 전개될 것이며, 궁극적으로 MIRIX는 **장기적 개인화, 신뢰할 수 있는 멀티턴 상호작용, 그리고 실세계 AI 애플리케이션의 기반**이 될 것으로 예상된다.[1]

***

### 참고 문헌 정리

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7d164e3-f314-42e6-87d6-ab4118e9d3ff/2507.07957v1.pdf)
[2](https://huggingface.co/blog/driaforall/mem-agent-blog)
[3](https://arxiv.org/abs/2505.19567)
[4](https://arxiv.org/abs/2410.06153)
[5](https://www.mongodb.com/company/blog/technical/why-multi-agent-systems-need-memory-engineering)
[6](https://arxiv.org/html/2503.21760)
[7](https://openreview.net/pdf?id=ZgQ0t3zYTQ)
[8](http://arxiv.org/pdf/2312.17259.pdf)
[9](https://www.sciencedirect.com/science/article/abs/pii/S0925231225008653)
[10](https://arxiv.org/abs/2504.01963)
[11](https://arxiv.org/abs/2505.13940)
[12](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd82705f44793b6f9ade5a669d0f0bf-Paper-Conference.pdf)
[13](https://ieeexplore.ieee.org/document/11232515/)
[14](https://sol.sbc.org.br/index.php/wesaac/article/view/37536)
[15](https://arxiv.org/abs/2504.20117)
[16](https://arxiv.org/abs/2409.11393)
[17](https://arxiv.org/abs/2511.00993)
[18](https://arxiv.org/abs/2408.09559)
[19](http://arxiv.org/pdf/2409.00872.pdf)
[20](https://arxiv.org/pdf/2502.12110.pdf)
[21](http://arxiv.org/pdf/2502.13843.pdf)
[22](http://arxiv.org/pdf/2304.13343.pdf)
[23](http://arxiv.org/pdf/2404.09982.pdf)
[24](http://arxiv.org/pdf/2408.09559.pdf)
[25](https://arxiv.org/html/2507.07957v1)
[26](https://openreview.net/forum?id=BryMFPQ4L6)
[27](https://research.ibm.com/blog/memory-augmented-LLMs)
[28](https://x2bee.tistory.com/412)
[29](https://www.emergentmind.com/topics/persistent-memory-for-llm-agents)
[30](https://supermemory.ai/research)
