# DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research

***

### **Executive Summary**

"DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research"는 2025년 11월에 발표된 혁신적인 논문으로, 장문형 심층 연구 작업을 위해 개발된 첫 번째 완전 오픈소스 모델입니다. 이 논문의 핵심 기여는 **Reinforcement Learning with Evolving Rubrics (RLER)**이라는 새로운 훈련 방법론을 제시하는 것으로, 기존의 정적(static)인 평가 기준을 동적으로 진화시키는 방식으로 심층 연구 모델을 훈련합니다.[1]

DR Tulu-8B는 기존 오픈소스 심층 연구 모델들을 13.7~53.4 포인트 상회하는 성능을 달성하며, 동시에 GPT-5와 Gemini 3 Pro를 포함한 독점 시스템과 비교할 수 있는 성능을 제공합니다. 특히 비용 효율성 측면에서 OpenAI Deep Research(쿼리당 $1.8)에 비해 거의 **3,000배 이상 저렴**($0.0019)하면서도 동등 이상의 성능을 보입니다.[2][1]

본 논문이 해결하는 핵심 문제는 다음과 같습니다: 기존 오픈소스 심층 연구 모델들은 쉽게 검증 가능한 단문형 QA 작업으로 훈련되었기 때문에, 실제 장문형, 개방형 심층 연구 작업으로의 일반화 성능이 제한적이었습니다.[1]

***

## **1. 해결하고자 하는 문제 (Problem Formulation)**

### **1.1 기존 접근법의 한계**

심층 연구(Deep Research, DR) 모델은 다단계 검색을 수행하여 장문형, 잘-속성화된(well-attributed) 답변을 생성하는 것을 목표합니다. 그러나 기존 오픈소스 심층 연구 모델들은 두 가지 근본적인 문제를 안고 있습니다:[1]

**문제 1: 검증 가능성의 제약**
- 기존 모델들은 **Reinforcement Learning with Verifiable Rewards (RLVR)**을 사용하여 단문형 QA 작업으로 훈련됩니다.[1]
- RLVR은 정답이 명확하게 검증 가능한 환경에서만 효과적입니다.
- 장문형 작업의 경우 "좋은 응답"의 정의가 복수이고 모호하므로 검증 가능한 보상을 정의하기 어렵습니다.[1]

**문제 2: 평가 기준의 정적 특성**
- 폐쇄책(closed-book) 루브릭: LM의 매개변수 지식만으로 생성되므로 세상의 동적 지식을 반영하지 못합니다.[1]
- 고정된 루브릭: 모델 훈련 중 새로이 탐색된 정보나 발생하는 새로운 행동 패턴을 반영할 수 없습니다.[1]

### **1.2 문제의 수학적 형식화**

논문은 심층 연구 작업을 다음과 같이 형식화합니다:[1]

$$S_i \leftarrow S_{i-1} + [a_i, \tau_i, o_i]$$

여기서:
- $S_i$: 시점 $i$에서의 상태(이전 맥락 + 최신 출력)
- $a_i$: 행동(think, tool, answer, cite 중 하나)
- $\tau_i$: 행동의 내용 또는 인수
- $o_i$: 도구 호출 시 반환되는 관찰값

각 도구 $T_k$는 쿼리 $q$와 옵션 인수를 받아 관찰값 $o = T_k(q)$를 반환합니다.[1]

루브릭 기반 보상 함수는 다음과 같이 정의됩니다:

$$S(x, y) = \sum_{k=1}^{K} \frac{w_{x,k} \cdot \text{Judge}(r_{x,k}, y)}{\sum_{k=1}^{K} w_{x,k}}, \quad w_{x,k} \in (-1, 1)$$

여기서:
- $r_{x,k}$: $k$번째 루브릭
- $\text{Judge}(r_{x,k}, y)$: 0, 0.5, 또는 1의 값을 반환
- $w_{x,k}$: 루브릭의 가중치(음수 가중치로 바람직하지 않은 특성 패널티 가능)[1]

***

## **2. 제안하는 방법 (Proposed Method: RLER)**

### **2.1 RLER의 핵심 개념**

**Reinforcement Learning with Evolving Rubrics (RLER)**의 기본 원칙은 다음과 같습니다:

> "훈련 과정 중 정책 모델과 함께 루브릭을 공진화(co-evolve)시킨다. 이를 통해 루브릭이 모델이 새로이 탐색한 정보를 통합하고, 정책에 대해 판별적이고 온정책(on-policy) 피드백을 제공하도록 한다."[1]

### **2.2 RLER의 세 가지 핵심 구성요소**

#### **A. 초기 검색 기반 루브릭 (Initial Search-Based Rubrics)**

각 질문 $x$에 대해, 먼저 인터넷 검색을 수행하여 관련 문맥을 검색합니다:[1]

$$R^{\text{persist}}(x) = G_{\text{rubric}}(x, \text{Search}(x))$$

여기서:
- $G_{\text{rubric}}$: 루브릭 생성 LM
- $\text{Search}(x)$: 질문 $x$로부터 검색된 문서들
- $R^{\text{persist}}(x) = \{R_1, R_2, \ldots, R_{K_s}\}$: 지속적 루브릭들

이는 폐쇄책 루브릭과 다르게 **실제 웹 지식으로 접지(ground)**되므로, 더 구체적이고 사실적인 평가 기준을 생성합니다.[1]

#### **B. 진화하는 루브릭 (Evolving Rubrics During Training)**

각 훈련 단계 $t$에서, 정책 모델로부터 $G$개의 롤아웃을 샘플링한 후:[1]

$$R^{\text{new}}(x) = G_{\text{rubric}}(x, \{y_i\}_{i=1}^{G}, R^{\text{persist}}(x) \cup R^{\text{active}}(x))$$

여기서:
- $\{y_i\}_{i=1}^{G}$: 현재 정책에서 샘플링한 응답들
- $R^{\text{new}}(x)$: 새로운 진화하는 루브릭 집합

생성되는 루브릭의 유형:[1]
1. **양성 루브릭**: 현재 정책이 새로이 탐색한 강점이나 정보 반영
2. **음성 루브릭**: 보상 해킹(reward hacking) 행동 억제 (예: 무의미한 코드 생성, 인용 조작)

예시에서 보면, 정책이 예상치 못하게 Python 코드를 생성하자, 진화하는 루브릭이 "무관한 코드 출력 제거"라는 음성 루브릭을 생성하여 이를 패널티 처리합니다.[1]

#### **C. 루브릭 버퍼 관리 (Rubric Buffer Management)**

무한정 증가하는 루브릭을 관리하기 위해, 다음 전략을 적용합니다:[1]

$$R^{\text{active}}(x) = \text{TopK}_{\text{std}}(R^{\text{new}}(x) \cup R^{\text{active}}(x), K_{\max})$$

여기서:
- 각 루브릭의 분산(표준편차)을 계산
- 분산이 0인 루브릭(판별력 없음) 제거
- 분산이 가장 높은 상위 $K_{\max}$개 루브릭만 보유

이는 계산 비용을 제어하면서도 **가장 정보가 풍부한 루브릭만 유지**합니다.[1]

### **2.3 RLER 훈련 알고리즘 (Algorithm 1)**

```
Algorithm: Reinforcement Learning with Evolving Rubrics (RLER)

Input: Dataset D, policy π_θ, rollout size G, max active rubrics K_max

1. for each prompt x ∈ D do
2.   Generate R^persist(x) ← G_rubric(x, Search(x))  // 초기 검색 기반 루브릭
3.   R^active(x) ← ∅
4.   
5.   for each training step t = 1, ..., T do
6.     R(x) ← R^persist(x) ∪ R^active(x)  // 합친 루브릭 집합
7.     Rollout with search: {y_i}_{i=1}^G ← π_θ(x)  // G개 롤아웃 생성
8.     Generate R^new(x) ← G_rubric(x, {y_i}_{i=1}^G, R(x))  // 진화하는 루브릭
9.     R^active(x) ← R^new(x) ∪ R^active(x)  // 활성 루브릭에 추가
10.    
11.    Compute rewards with R^persist(x) ∪ R^active(x) and update GRPO
12.    Compute std of rewards per rubric
13.    For R^active(x), remove rubrics with 0 std
14.    Keep top-K_max with highest std  // 루브릭 버퍼 관리
15.  end for
16. end for
```


### **2.4 보조 보상 (Auxiliary Rewards)**

루브릭 기반 보상 외에 세 가지 보조 보상을 추가합니다:[1]

$$R_{\text{total}} = R_{\text{rubric}} + \alpha_f \cdot R_{\text{format}} + \alpha_s \cdot R_{\text{search}} + \alpha_c \cdot R_{\text{citation}}$$

- $R_{\text{format}}$: 올바른 형식 준수 장려
- $R_{\text{search}}$: 도구 사용 장려
- $R_{\text{citation}}$: 고품질 인용 제공 장려

***

## **3. 모델 구조 (Architecture)**

### **3.1 기본 모델 및 Supervised Fine-Tuning (SFT)**

#### **기본 모델**
- **Backbone**: Qwen3-8B[1]
- **선택 이유**: 오픈소스이면서 효율적인 크기와 능력 제공

#### **SFT 데이터 구성**
논문은 16K의 SFT 데이터를 수집했습니다:[1]

1. **장문형 쿼리 (11K)**
   - SearchArena: 24K 실시간 사용자-보조 상호작용
   - OpenScholar: 55K 과학 연구 기반 쿼리
   - LM 기반 필터링으로 고품질 프롬프트만 선별

2. **단문형 QA (5K)**
   - HotpotQA, TaskCraft, WebWalker-Silver, MegaScience
   - 합성 프롬프트 (PopQA 기반)
   - 목표: 단일 작업에 과적합 방지

#### **궤적(Trajectory) 생성**
- **LM**: GPT-5를 사용하여 엔드-투-엔드 궤적 생성[1]
- **도구**: 일반 웹 검색, 논문 검색, 웹 브라우징
- **거부 샘플링**: 형식 준수 + 단문형 정답 일치 검증

### **3.2 온라인 RL 훈련 (Online RL with Asynchronous Tool Calls)**

#### **훈련 데이터**
- 장문형 질문만 사용: ~5K (SearchArena + OpenScholar) + ~4K (RaR)
- 총 약 9K 질문

#### **GRPO 알고리즘**
DR Tulu는 **Group Relative Policy Optimization (GRPO)**을 사용합니다:[3][1]

GRPO의 핵심:
- PPO와 달리, 별도의 가치 함수(critic) 없이 그룹 평균을 기준으로 이점 계산
- 각 질문에 대해 $G$개의 완성을 샘플링하고, 그룹 내에서 상대 비교

$$\text{Advantage}_{i,g} = R_{i,g} - \text{mean}(R_{i,:})$$

여기서 $R_{i,g}$는 샘플 $g$의 보상, mean은 그룹 평균입니다.[1]

#### **비동기 도구 호출**
기존 방식: 배치 전체 생성 완료 → 도구 호출 순차 처리
DR Tulu 방식:[1]
- 도구 호출 직후 해당 생성 요청을 대기 상태로
- 추론 엔진이 다른 응답 생성 계속 수행
- 도구 응답 수신 시 재개

**결과**: 생성과 도구 호출이 겹치면서 훈련 시간 단축

### **3.3 dr-agent-lib: 심층 연구용 기반 구조**

논문은 새로운 **MCP 기반 에이전트 라이브러리**를 제공합니다:[1]

**주요 기능:**
1. **통합 도구 백엔드**
   - 구글 검색(googlesearch): 상위 웹 스니펫
   - 웹 브라우징(webbrowse): 크롤링된 페이지 텍스트
   - 논문 검색(papersearch): 오픈 액세스 논문 검색

2. **동시성 최적화**
   - 글로벌 캐시: 반복 쿼리 캐싱
   - 비동기 프로세스 잠금: API 요청 속도 제한 관리

3. **유연한 프롬프트 계층**
   - 다양한 검색 워크플로우 구성 가능
   - 도구별 프롬프트 및 호출 설정 세밀 제어

***

## **4. 성능 향상 및 벤치마크 결과**

### **4.1 장문형 심층 연구 벤치마크**

논문은 4개의 주요 벤치마크에서 평가합니다:[1]

| 벤치마크 | 도메인 | 특징 |
|---------|---------|------|
| **ScholarQA-CSv2** | 과학 | 학술 문헌 종합 |
| **HealthBench** | 의료 | 의료 질문, 유해 응답 패널티 포함 |
| **ResearchQA** | 과학 | 최신 과학 문헌 종합 |
| **DeepResearchBench** | 일반 영역 | 다양한 일반 질문 |

### **4.2 주요 결과**

#### **개요 성능 비교**

| 모델 | 파라미터 | SQAv2 | HealthBench | ResearchQA | DRB | 평균 |
|------|---------|-------|------------|-----------|-----|------|
| **오픈소스 모델들** |
| Search-R1 | 7B | 22.2 | -0.1 | 27.9 | 9.5 | 14.9 |
| WebThinker | 32B | 32.9 | 11.1 | 48.6 | 23.3 | 28.9 |
| Tongyi DeepResearch | 30B | 46.5 | 46.2 | 66.7 | 40.6 | 50.0 |
| **DR Tulu-8B (제안)** | **8B** | **86.8** | **50.2** | **74.3** | **43.4** | **63.7** |
| **폐쇄소스 모델들** |
| Gemini 3 Pro Search | - | 69.8 | 38.0 | 74.3 | 46.3 | 57.0 |
| OpenAI Deep Research | - | 79.6 | 53.8 | 79.2 | 46.9 | 64.9 |

[2][1]

**주요 발견:**
- DR Tulu-8B는 **모든 오픈소스 모델을 큰 차이로 상회**
- ResearchQA와 HealthBench에서 폐쇄소스 모델도 추월
- 단 8B 파라미터로 30B 모델(Tongyi)보다 13-27 포인트 우수

#### **상세 성능 분석 (SQAv2 & DeepResearchBench)**

| 지표 | SQAv2 | DRB |
|------|-------|-----|
| 루브릭 점수 | 89.6 | 41.7 |
| 답변 정확도 | 95.4 | 41.8 |
| 인용 정밀도 | 88.6 | - |
| 인용 재현도 | **73.7** | - |
| 종합성 | - | 48.2 |
| 깊이 | - | 41.3 |

[1]

**특별히 주목할 점**: 
- 인용 정밀도 88.6%: 정책이 인용한 출처가 실제로 주장을 지지
- 인용 재현도 73.7%: 인용 가능한 모든 주장의 73.7%에 인용 제공

### **4.3 비용 효율성 비교**

| 시스템 | API 비용 | 모델 비용 | 총계/쿼리 | 상대 비용 |
|---------|---------|---------|---------|---------|
| OpenAI Deep Research | - | $1.80 | **$1.80** | **947배** 더 비쌈 |
| Ai2 ScholarQA (Claude) | - | $1.30 | $1.30 | 684배 더 비쌈 |
| Tongyi DeepResearch | $0.032 | - | $0.032 | 16.8배 더 비쌈 |
| WebThinker-32B report | $0.015 | - | $0.015 | 7.9배 더 비쌈 |
| **DR Tulu-8B** | **$0.00008** | **$0.0018** | **$0.0019** | **기준선** |

[2][1]

### **4.4 RLER의 성능 향상**

**SFT vs. RL (RLER 적용) 비교:**

| 지표 | SQAv2 | HealthBench | ResearchQA | DRB | 향상도 |
|------|-------|-----------|-----------|-----|--------|
| 루브릭 (SFT) | 81.4 | 38.1 | 68.5 | 39.0 | - |
| 루브릭 (RL) | 89.6 | 39.3 | 68.7 | 40.2 | +8.2 |
| 인용 정밀도 (SFT) | 65.3 | - | - | - | - |
| 인용 정밀도 (RL) | 88.6 | - | - | - | +23.3 |
| 인용 재현도 (SFT) | 51.6 | - | - | - | - |
| 인용 재현도 (RL) | 73.7 | - | - | - | +22.1 |

[1]

**해석:**
- RLER이 각 벤치마크에서 **4.4~4.5 포인트 향상** 제공
- 인용 정밀도와 재현도에서 **특히 큰 개선** (23~22 포인트)
- 이는 진화하는 루브릭이 정확한 속성화(attribution)를 효과적으로 강화함을 시사

***

## **5. 모델의 일반화 성능 향상 가능성 (Generalization & Domain Transfer)**

### **5.1 단문형 QA 성능 (일반화성 검증)**

RLER이 장문형 작업으로만 훈련되었음에도 불구하고, 단문형 QA에서도 성능이 향상됩니다:[1]

| 벤치마크 | Qwen3-8B | DR Tulu SFT | DR Tulu RL | 향상도 |
|---------|---------|-----------|----------|--------|
| SimpleQA | 52.6 | 75.5 | 80.1 | +27.5 |
| 2Wiki | 18.9 | 66.5 | 68.0 | +49.1 |
| WebWalker | 8.8 | 31.9 | 39.1 | +30.3 |
| **평균** | 26.8 | 58.0 | 62.4 | +35.6 |

[1]

**일반화 메커니즘:**
1. SFT 데이터에 단문형 QA 포함 (5K/16K, 31%)
2. RLER 훈련 중 모델이 적응적으로 응답 길이 조절 학습
3. 긴 설명이 필요 없는 질문에 대해서도 효과적으로 단답 생성

### **5.2 도메인 밖(Out-of-Domain) 적응: 유전자 질환 분석**

새로운 **GeneticDiseasesQA** 벤치마크를 구축하여 도메인 외 실제 의료 작업에서 평가:[1]

**작업**: 임상 유전학자가 사용하는 유전 변이 해석
- 47개 질문, 24개 질병 원인 유전 변이
- 분자 메커니즘, 치료 접근법 추론 필요

**결과:**

| 지표 | 최종 답변 | 증거 품질 | 증거 종합 | 증거 지지도 |
|------|---------|---------|---------|---------|
| DR Tulu-8B | 33 | 38 | 50 | 67 |
| Qwen3-8B | 21 | 24 | 32 | 58 |
| Gemini 3 Pro | 31 | 36 | 43 | 56 |
| GPT-5 | 35 | 44 | 51 | 69 |
| Claude Sonnet | - | - | - | - |

[1]

**의의:**
- DR Tulu는 훈련 데이터에 없던 전문 임상 작업으로 **효과적으로 전이**
- 속성화(attribution)와 증거 합성 측면에서 특히 강점 (67% 증거 지지도)
- 폐쇄소스 모델과 비슷하거나 우수한 성능

### **5.3 일반화성 제약 분석**

#### **훈련-테스트 불일치 문제**

논문은 중요한 발견을 보고합니다:[1]

> "훈련 중 보상이 높은 모델이 반드시 다운스트림 평가에서 최고 성능을 내지는 않음"

**원인:**
1. 훈련 판사(GPT-4.1-mini) ≠ 평가 판사(벤치마크별 다름)
2. 외부 벤치마크의 루브릭이 훈련 루브릭과 강조점 상이
3. 보상 해킹: 모델이 훈련 판사의 편향 학습 (예: 단순히 길이만 증가)

**완화 전략:**
- 진화하는 루브릭이 음성 루브릭을 통해 일부 해킹 방지
- 다양한 SFT 데이터 혼합으로 과적합 완화
- 더 나은 일반화를 위해서는 평가 기준과 훈련 목표의 정렬 필요

### **5.4 SFT 데이터 조성의 중요성**

광범위한 절제(ablation) 연구를 통해:[1]

| 구성 | SQAv2 | HealthBench | ResearchQA | DRB |
|------|-------|------------|-----------|-----|
| 장문형만 | 81.2 | 40.7 | 71.2 | 42.1 |
| 단문형만 | 77.8 | 38.5 | 66.3 | 38.9 |
| **혼합 (최적)** | **82.3** | **39.3** | **67.8** | **39.0** |

[1]

**결론:**
- 장문형만으로 훈련하면 단문형 일반화 성능 저하
- 단문형만으로 훈련하면 장문형 성능 급격히 저하
- **혼합 구성이 전체 일반화 성능 극대화**

***

## **6. 한계 및 과제 (Limitations)**

### **6.1 기술적 한계**

#### **1. 루브릭 생성 비용**
- 각 훈련 단계마다 새 루브릭 생성 필요
- 훈련 비용 상당 증가 (총 9,700 GPU 시간, 25일)
- 루브릭 생성 LM(GPT-4.1)의 비용 의존성

#### **2. API 속도 제한**
- 도구 호출이 훈련 병목
- "증가 계산이 RL 훈련 속도 개선 못함" (논문 기술)
- 비동기 호출로 부분적 해결했지만 완전 해결 아님

#### **3. 훈련-평가 불일치**
- 외부 벤치마크의 평가 기준과 훈련 루브릭 간 격차
- 보상 해킹 위험: 모델이 훈련 판사의 편향 학습

### **6.2 방법론적 한계**

#### **1. 루브릭의 질 의존성**
- 루브릭 생성 LM의 능력에 크게 의존
- 초기 검색 질과 진화 프롬프트 설계에 민감

#### **2. 지식 기반 작업으로 제한**
- 텍스트 기반 심층 연구에 초점
- 구조화된 데이터, 이미지, 모달리티 통합 미흡

#### **3. 도메인 적응의 한계**
- GeneticDiseasesQA에서 최고 성능 모델(GPT-5)에 비해 약간 뒤짐
- 매우 전문화된 영역에서는 추가 훈련 또는 특화 필요 가능

### **6.3 평가 메트릭의 한계**

- 대부분 LLM 판사 기반 평가 (자동 평가 편향 위험)
- 일부 벤치마크는 샘플 크기 작음 (100개 미만)
- 인간 평가 비용으로 광범위 검증 미흡

***

## **7. 논문의 앞으로의 연구에 미치는 영향**

### **7.1 RLER 프레임워크의 광범위 적용 가능성**

#### **1. 다른 장문형 생성 작업**
- 요약 (Summarization)
- 보고서 생성 (Report Generation)
- 기술 문서 작성 (Technical Documentation)

RLER의 핵심 원칙(동적 루브릭 + 지식 접지)은 모든 검증 어려운 개방형 작업에 전이 가능합니다.[1]

#### **2. 다중 모달 심층 연구**
논문에서 언급한 미래 방향:[1]

> "많은 과학 분야는 구조화된, 도메인 특화 도구에 의존 (예: 게놈 수열, 분자 구조, 트랜스크립토믹스 등). 이런 특화 데이터 소스를 훈련에 통합하거나, 미학습 도구를 적시(just-in-time)에 사용하도록 모델을 훈련하는 것이 자연스러운 다음 단계"

### **7.2 보상 함수 설계 패러다임의 변화**

#### **기존 패러다임**
- **정적 루브릭**: 사전에 모든 평가 기준 정의
- **한계**: 모호한 작업에 불충분

#### **새 패러다임 (RLER)**
- **동적 루브릭**: 훈련 중 진화
- **장점**: 새로운 행동 패턴 자동 감지 및 대응

이는 **보상 해킹 완화의 새 방향**을 제시합니다. 기존 보상 해킹 방지 연구:[4][1]
- 펼쳐진 검증자(unfolded verifiers)
- 적대적 훈련
- 구조화된 루브릭

RLER은 **검증자를 적응적으로 개선**하는 상호보강적 접근입니다.[1]

### **7.3 오픈소스 심층 연구 생태계 구축**

#### **공개 자산**
- 모델 체크포인트: Hugging Face
- 훈련 코드: GitHub (open-instruct 기반)
- 평가 데이터: 4개 벤치마크
- 인프라: dr-agent-lib (MCP 기반)

**영향:**
- 연구자들이 쉽게 커스터마이징 가능한 기반 제공
- 보유된 데이터에 특화된 모델 훈련 가능
- 데이터 프라이버시 보존 (로컬 배포)

### **7.4 관련 연구 분야에의 영향**

#### **1. 평가 기준 생성 연구**
- EvalAgent (Wadhwa et al., 2025): 웹에서 암묵적 평가 기준 발견
- Rubrics as Rewards (Gunjal et al., 2025): 인스턴스별 루브릭 기반 보상

RLER은 이들을 **훈련 중 통합 및 진화**시킵니다.[1]

#### **2. RLVR 확장**
- Crossing the Reward Bridge (Liu et al., 2025): 검증 가능 보상 도메인 확장
- DR Tulu는 **검증 불가능한 작업**으로 RLVR 개념 확장[1]

#### **3. 다목표 강화학습**
- MO-GRPO (Peng et al., 2025): 다중 목표 GRPO
- DR Tulu의 다중 보상 구성(루브릭 + 형식 + 인용)이 선례 제시

***

## **8. 앞으로 연구 시 고려할 점**

### **8.1 즉시 실행 가능한 개선 방향**

#### **1. 루브릭 생성 최적화**
- **현재**: 매 단계마다 새 루브릭 생성 (비용 높음)
- **개선안**:
  - 루브릭 갱신 빈도 최적화 (e.g., 매 5 단계마다)
  - 더 경량의 루브릭 생성 모델 사용 (예: Qwen3-8B 자체)
  - 루브릭 캐싱 및 재사용 메커니즘

$$\text{Cost Reduction} = 1 - \frac{\text{Update Frequency}}{\text{Training Steps}} \times \text{Overhead}$$

#### **2. 도구 효율성 개선**
- **현재**: API 속도 제한이 병목
- **개선안**:
  - 로컬 검색 엔진 적분 (e.g., Elasticsearch)
  - 도구 응답 캐싱 확대
  - 배치 API 호출 최적화

#### **3. 훈련-평가 정렬 강화**
- **메커니즘**: 다양한 외부 벤치마크의 루브릭을 훈련 초기에 포함
- **구현**: "평가 루브릭 주입" 단계 추가
  
$$R_{\text{eval-aligned}}(x) = \alpha \cdot R_{\text{evolving}}(x) + (1-\alpha) \cdot R_{\text{external}}(x)$$

### **8.2 중기 연구 방향 (1~2년)**

#### **1. 다중 모달 확장**
과학 분야의 복합 데이터 처리:
- **텍스트 + 표/그래프**: 표 해석 및 시각화 분석
- **텍스트 + 이미지**: 의료 영상 해석 (의료 심층 연구)
- **텍스트 + 구조**: 분자 구조 분석 (화학/생물학)

```
G_rubric(x, {y_i}^G, [text, table, image, structure])
```

#### **2. 장기 맥락 처리**
- 현재: 토큰 한계로 제한된 검색 결과 통합
- 개선: 계층적 정보 통합 및 압축
- 기법: 검색 결과의 동적 요약 및 선택

#### **3. 설명 가능성 강화**
- 각 루브릭의 역할 추적
- 루브릭 진화 과정 시각화
- 모델 결정의 증명 가능한 추적성

### **8.3 장기 전략 (2년 이상)**

#### **1. 인간-in-the-loop 진화**
```
Human feedback → Rubric refinement → Policy update
```
- 사용자 선호도 직접 루브릭에 반영
- 도메인 전문가 피드백 통합

#### **2. 전이 학습 프레임워크**
- 기본 심층 연구 능력을 한 번 훈련
- 다운스트림 전문 영역에 빠른 적응
- 추가 RL 단계 없이 프롬프트 엔지니어링으로 조정

#### **3. 분산 훈련 및 연합 학습**
- 여러 조직이 협력하여 모델 개선
- 각 조직의 데이터 프라이버시 보호
- 글로벌 루브릭 공유 및 지역화

### **8.4 방법론적 성숙화**

#### **1. 이론적 분석**
현재 논문이 부족한 부분:
- RLER 수렴성 증명
- 루브릭 버퍼 크기의 최적 선택
- 진화 속도 vs. 성능의 트레이드오프 분석

$$\text{Convergence Rate} = f(K_{\max}, \text{Update Frequency}, \text{Rubric Quality})$$

#### **2. 일반화 성능 이론**
- 훈련-테스트 불일치 정량화
- 최악의 경우 성능 보장
- PAC 학습 프레임워크에서의 RLER 분석

#### **3. 비교 연구**
- RLER vs. 다른 동적 평가 방법 (e.g., 대조적 롤아웃)
- 다양한 GRPO 변형 (Hybrid GRPO, DrGRPO 등)과 비교
- 저자원 환경에서의 성능

***

## **9. 결론 및 종합 평가**

### **9.1 DR Tulu의 주요 기여 요약**

| 차원 | 기여 |
|------|------|
| **방법론** | RLER: 검증 불가능한 작업에 RL 확장 |
| **성능** | 8B로 30B 모델 능가, 폐쇄소스와 경쟁 가능 |
| **효율성** | 쿼리당 비용 947배 저감 |
| **개방성** | 완전 오픈소스 (코드, 데이터, 모델) |
| **신뢰성** | 높은 인용 정밀도 (88.6%) 및 재현도 (73.7%) |
| **일반화** | 도메인 외 작업에 효과적 전이 |

### **9.2 RLER의 과학적 의의**

**기본 과학 질문**: "정적 검증자 없이 복잡한 작업을 학습할 수 있는가?"

**답**: "예. 검증자를 동적으로 진화시킴으로써 가능."

이는 강화학습 이론에 새로운 관점을 제시합니다:
- **보상 함수 = 고정 대상**이라는 기존 가정 도전
- **검증자 자체도 학습의 대상**이 될 수 있음을 시현
- 공진화(co-evolution) 패러다임의 효과성 입증

### **9.3 실무적 영향**

#### **현재 (2025년~)**
- 개인 또는 소규모 팀이 비용 효율적으로 심층 연구 시스템 구축 가능
- 기존 AI 에이전트에 심층 연구 능력 통합
- 데이터 프라이버시 보존하며 로컬 배포 가능

#### **중기 (1~2년)**
- 의료, 법률, 과학 분야 도메인 특화 버전 등장
- 다중 모달 심층 연구 시스템으로 확대
- 학술 및 기업 연구 표준 도구로 채택

#### **장기 (2년 이상)**
- 자율적 과학 발견 에이전트의 핵심 기술로 확립
- 새로운 지식 합성 및 가설 생성에 활용
- 인간-AI 협업 연구의 표준 플랫폼

### **9.4 열린 질문 및 미래 연구**

1. **확장성**: RLER이 매우 큰 모델(100B+)에 효과적인가?
2. **안정성**: 장시간 훈련 중 루브릭 드리프트(drift)는 없는가?
3. **이전성**: 한 도메인에서 훈련한 루브릭을 다른 도메인에 적용 가능한가?
4. **견고성**: 악의적 프롬프트나 분포 외 입력에 대한 견고성은?
5. **윤리**: 자동 심층 연구가 거짓 정보 확산에 악용될 수 있는가?

***

## **참고 문헌 및 최신 연구**

### **직접 인용 논문**
 Shao et al., "DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research," arXiv:2511.19399, November 2025[1]

### **관련 최신 연구 (2025년)**

**GRPO 및 강화학습:**
-  "Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning" (2025)[3]
-  "MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization" (2025)[5]
-  "CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models" (2025)[6]
-  "Hybrid Group Relative Policy Optimization: A Multi-Sample Approach" (2025)[7]
-  "Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training" (2025)[8]

**평가 및 루브릭:**
-  "Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains" (2025)[9]
-  "Reward Hacking in RLVR Systems" (Emergent Mind, 2025)[4]
-  "An Empirical Study of Evaluating Long-form Question Answering" (2025)[10]
-  "Localizing and Mitigating Errors in Long-form Question Answering" (2025)[11]

**장문형 생성:**
-  "ELI5: Long Form Question Answering" (Meta AI)[12]
-  "Improving Attributed Long-form Question Answering with Intent Awareness" (2025)[13]

***

## **최종 정리: DR Tulu의 위치 및 의미**

DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research는:

1. **방법론적으로**: 강화학습의 경계를 확장하여 검증 불가능한 복잡한 작업 해결
2. **기술적으로**: 소형 모델(8B)이 대규모 모델 성능 달성
3. **경제적으로**: AI 민주화를 통해 모든 규모의 조직이 심층 연구 능력 확보
4. **과학적으로**: 심층 연구 벤치마크의 새로운 상향 기준 설정
5. **실무적으로**: 의료, 법률, 과학 등 지식 집약적 분야의 의사 결정 지원 도구

이 논문은 단순한 기술 혁신을 넘어, **AI 시스템이 복잡한 실세계 문제에 어떻게 적응하고 개선될 수 있는가**에 대한 근본적인 질문에 답합니다. RLER의 개념과 방법론은 향후 5~10년 AI 연구의 중요한 축이 될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/50585313-3503-4821-8337-50273cf603a0/2511.19399v2.pdf)
[2](https://arxiv.org/abs/2511.19399)
[3](https://arxiv.org/abs/2510.19807)
[4](https://www.emergentmind.com/topics/reward-hacking-in-reinforcement-learning-with-verifiable-rewards-rlvr)
[5](https://arxiv.org/abs/2509.22047)
[6](https://arxiv.org/abs/2503.22342)
[7](https://arxiv.org/abs/2502.01652)
[8](https://arxiv.org/abs/2505.22257)
[9](https://arxiv.org/pdf/2503.23829.pdf)
[10](https://arxiv.org/abs/2504.18413)
[11](https://aclanthology.org/2025.findings-acl.1049/)
[12](https://ai.meta.com/research/publications/eli5-long-form-question-answering/)
[13](https://openreview.net/forum?id=fRCm5c8x0j)
[14](http://arxiv.org/pdf/1708.05866v1.pdf)
[15](http://arxiv.org/pdf/2503.09270.pdf)
[16](https://arxiv.org/pdf/2411.18892.pdf)
[17](https://arxiv.org/abs/2412.05265)
[18](http://arxiv.org/pdf/2405.10214.pdf)
[19](https://arxiv.org/pdf/2410.15127v1.pdf)
[20](https://arxiv.org/pdf/2107.03015.pdf)
[21](https://huggingface.co/papers/2511.19399)
[22](https://arxiv.org/html/2509.15557v1)
[23](https://www.datocms-assets.com/64837/1763496622-dr_tulu_draft.pdf)
[24](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
[25](https://www.youtube.com/watch?v=_rNVtZOZ-MM)
[26](https://arxiv.org/abs/2503.03797)
[27](https://www.semanticscholar.org/paper/37180396829cdf10a94bb5bb22858afbdb818294)
[28](https://arxiv.org/abs/2510.08607)
[29](https://dl.acm.org/doi/10.1145/3733006.3733021)
[30](https://arxiv.org/abs/2504.02407)
[31](http://arxiv.org/pdf/2503.21819.pdf)
[32](https://arxiv.org/pdf/2503.06639.pdf)
[33](https://arxiv.org/pdf/2502.01652.pdf)
[34](https://arxiv.org/pdf/2503.22342.pdf)
[35](http://arxiv.org/pdf/2503.03797.pdf)
[36](https://arxiv.org/pdf/2405.20304.pdf)
[37](https://arxiv.org/html/2503.15952v1)
[38](https://arxiv.org/pdf/2501.03262.pdf)
[39](https://verl.readthedocs.io/en/latest/algo/grpo.html)
[40](https://milvus.io/ai-quick-reference/what-are-precision-and-recall-in-ir)
[41](https://www.ijcai.org/proceedings/2025/1198.pdf)
[42](https://velog.io/@choseon94/Understanding-GRPO)
[43](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)
[44](https://aclanthology.org/2025.findings-emnlp.1085.pdf)
[45](https://aiengineering.academy/LLM/TheoryBehindFinetuning/GRPO/)
[46](https://en.wikipedia.org/wiki/Precision_and_recall)
[47](https://openreview.net/pdf/14d4ad1a5cbb8e65dcc52325ec93b1aedcc931bc.pdf)
[48](https://jaylala.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-with-Python-GRPO%EB%9E%80-Group-Relative-Policy-Optimization)
[49](https://smcho1201.tistory.com/137)
