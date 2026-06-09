
# Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses 

> **📌 논문 정보**
> - **제목:** Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses
> - **저자:** Pengcheng Jiang 외 7명
> - **arXiv:** [arXiv:2606.02373](https://arxiv.org/abs/2606.02373) (제출일: 2026년 6월 1일)
> - **코드:** [github.com/pat-jj/harness-1](https://github.com/pat-jj/harness-1)
> - **모델 가중치:** [huggingface.co/pat-jj/harness-1](https://huggingface.co/pat-jj/harness-1)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장 (Core Claim)

기존 검색 에이전트는 점점 길어지는 트랜스크립트(transcript) 위에서 정책(policy)을 학습하는 방식으로, 모델이 검색 방법을 결정하는 동시에 무엇을 보았는지 기억하고, 어떤 증거가 유용한지 판단하며, 어떤 제약이 남아 있는지 추적해야 한다. 논문은 이 방식이 정책 내부에 너무 많은 루틴적 상태 관리를 떠넘기며, RL이 회복 가능한 사무적 기록(bookkeeping)과 의미 있는 검색 결정 모두를 동시에 최적화해야 하는 비효율을 초래한다고 주장한다.

**핵심 제안:** Harness-1은 검색 에이전트에서 상태 관리와 정책 학습을 분리하여, 작업 메모리(working memory)를 모델 내부가 아닌 환경(environment)으로 외부화(externalize)한다.

### 주요 기여 (Contributions)

| 기여 항목 | 설명 |
|---|---|
| **State-Externalizing Harness** | 환경 측에서 상태를 유지하는 구조적 하네스 설계 |
| **Harness-1 모델** | 20B 파라미터 오픈 검색 서브에이전트 |
| **RL 학습 프레임워크** | 구조화된 검색 워크스페이스 위에서의 RL 훈련 |
| **전이 일반화 증거** | 훈련 도메인 외 벤치마크에서 강한 전이 성능 |
| **오픈소스 공개** | 가중치, 코드, 하네스 전체 공개 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상, 한계

### 2.1 해결하고자 하는 문제 (Problem Statement)

기존 에이전트 설명 방식은 에이전트의 가시적 행동만을 포착할 뿐, 에피소드 진행 과정에서 반드시 구축되어야 하는 검색 상태(search state)는 포착하지 못한다. 성공적인 멀티턴 검색 에피소드는 에이전트가 어떤 문서를 보았는지, 어떤 후보가 보존할 가치가 있는지, 어떤 제약이 미해결인지, 어떤 엔티티가 별개의 증거들을 연결하는지, 어떤 주장이 실제로 원문에 대조 검증되었는지를 기억해야 한다. 이 구분은 강화학습에서 특히 중요해진다.

이 접근법은 에이전틱 RL의 근본적 비효율성, 즉 모델이 추론과 회복 가능한 행정적 오버헤드 모두를 최적화하도록 강요받는 문제를 해결한다.

RL의 관점에서는 더욱 문제가 심화된다. 최종 보상은 에피소드가 성공했는지 여부만 알려줄 뿐, 왜 실패했는지는 알려주지 않는다. 나쁜 검색이었는가? 증거를 잊어버렸는가? 검증이 누락되었는가? 큐레이션이 부실했는가? 아니면 에이전트가 이미 본 것을 추적하지 못했는가?

---

### 2.2 제안하는 방법 (Proposed Method)

#### 2.2.1 State-Externalizing Harness 구조

하네스는 후보 풀(candidate pool), 중요도 태그가 붙은 큐레이션 세트, 컴팩트한 증거 링크, 검증 기록, 압축 및 중복 제거된 관측, 예산 인식 컨텍스트 렌더링 등의 환경 측 작업 메모리를 유지한다.

정책(policy)은 의미론적 결정들, 즉 무엇을 검색할지, 어떤 문서를 보존하거나 폐기할지, 무엇을 검증할지, 언제 멈출지를 보존한다.

구체적으로, 하네스는 후보 문서, 큐레이션된 증거, 중요도 태그, 검색 이력, 증거 링크, 검증 기록, 중복 제거/압축, 컨텍스트 예산 마커로 구성된 작업 메모리를 유지한다. 에이전트는 단순히 검색 박스와 대화하는 것이 아니라, 하나의 워크스페이스를 조작하는 것이다.

이 설계를 논문에서는 **"Stateful Cognitive Offloading(상태적 인지 오프로딩)"** 이라 명명한다.

이 설정으로 인해 RL은 모델이 비구조적인 트랜스크립트를 처음부터 관리하도록 가르칠 필요가 없다. 대신, 구조화된 검색 워크스페이스 위에서 운용되도록 훈련한다.

#### 2.2.2 훈련 파이프라인

훈련은 두 단계로 구성된다:

**① Supervised Fine-Tuning (SFT)**

단일 교사 모델인 GPT-5.4가 전체 하네스 내부에서 라이브로 실행되며, 필터링 후 899개의 트라젝토리가 SFT에 사용된다. 모델은 rank 32의 LoRA로 3 에포크 학습되며, step-550 체크포인트가 RL의 초기화에 사용된다.

**② Reinforcement Learning (RL)**

RL은 온-폴리시 CISPO 알고리즘을 사용하며, 40턴 상한 및 터미널(terminal) 전용 보상 구조를 가진다. SEC 쿼리만을 대상으로 학습하며, 동일한 보상을 가진 그룹은 그래디언트에서 제외된다.

#### 2.2.3 보상 함수 (Reward Function)

논문에서 보상은 두 가지 핵심 요소를 분리한다:

보상은 발견(discovery)과 선택(selection)을 분리하며, 도구 다양성 보너스(tool-diversity bonus)도 추가한다. 이 보너스가 없으면, 에이전트는 반복 검색으로 붕괴(collapse)한다.

논문에서 언급된 보상 구성을 수식으로 표현하면 다음과 같습니다:

$$R = R_{\text{curated}} + \lambda \cdot R_{\text{diversity}}$$

여기서:
- $R_{\text{curated}}$: 큐레이션된 문서 집합에서의 관련 문서 재현율(recall) 기반 보상 (발견 + 선택)
- $R_{\text{diversity}}$: 사용된 도구의 다양성에 대한 보너스 보상
- $\lambda$: 다양성 보너스의 가중치

이 보너스 없이는 큐레이션 재현율이 약 0.53에서 정체하고, 보너스를 추가하면 다양성이 안정화되며 재현율이 약 0.60에 도달한다.

**큐레이션 초기화 과정:**

첫 번째 성공적인 검색은 재랭킹된 8개 문서를 공정한 중요도로 큐레이션 세트에 자동으로 시드(seed)하며, 이후 정책은 강한 문서를 승격하고 약한 문서를 제거한다.

#### 2.2.4 세 가지 훈련 가능성 조건 (Trainability Requirements)

논문은 세 가지 훈련 가능성 요건을 제시한다: 웜-스타트 큐레이션(warm-started curation), 컴팩트 파생 상태 렌더링(compact derived-state rendering), 다양성 보존 인센티브(diversity-preserving incentives).

---

### 2.3 모델 구조 (Model Architecture)

| 항목 | 세부 내용 |
|---|---|
| **기반 모델** | gpt-oss-20B (20B 파라미터) |
| **역할** | Retrieval Subagent (답변 생성 ❌, 문서 검색 ✅) |
| **학습 방식** | SFT (LoRA rank-32) → RL (CISPO) |
| **하네스 구성** | 후보 풀 / 큐레이션 세트 / 중요도 태그 / 증거 링크 / 검증 기록 |
| **출력** | 다운스트림 답변 모델에 전달할 큐레이션 문서 집합 (최대 30개) |
| **연구 기관** | UIUC, UC Berkeley, Chroma |

Harness-1은 다운스트림 답변 모델을 위한 큐레이션 문서 세트를 반환한다. 질문에 직접 답변하지 않는다.

검색 에이전트에서 모델은 전체 학습 시스템이 아니다. 하네스 — 메모리 레이아웃, 액션 공간, 큐레이션 인터페이스, 검증 기록, 컨텍스트 렌더링 — 이 모두가 RL이 사용하도록 학습하는 대상의 일부이다.

---

### 2.4 성능 향상 (Performance Results)

Harness-1은 평가된 오픈 검색 서브에이전트 중 가장 강력하며, 평균 큐레이션 재현율 0.730을 달성하고 다음으로 강한 오픈 서브에이전트인 Tongyi DeepResearch 30B를 +11.4 포인트 차이로 능가한다.

프런티어 검색기들 중에서는 Opus-4.6만이 평균적으로 더 높은 점수를 기록한다.

검색 성능 향상은 다운스트림 답변 품질로도 이어진다.

| 비교 대상 | 평균 큐레이션 재현율 |
|---|---|
| **Harness-1 (20B)** | **0.730** |
| Tongyi DeepResearch (30B) | 0.616 (추정, +11.4p 차) |
| Opus-4.6 | Harness-1보다 약간 높음 |
| GPT-5.4 | Harness-1보다 낮음 |

Harness-1은 899개의 필터링된 SFT 트라젝토리와 3,453개의 쿼리로 RL을 수행한다.

---

### 2.5 한계 (Limitations)

1. **단일 도메인 RL 훈련:** RL은 SEC 쿼리만을 대상으로 학습된다, 즉 매우 제한된 도메인에서의 RL 훈련이다.

2. **하네스 의존성:** Ablation 실험에서 하네스 메커니즘을 제거하면 에이전트 행동이 변화하고 재현율이 하락한다는 것은, 모델 자체의 능력보다 하네스 구조에 대한 의존도가 높음을 의미한다.

3. **RL 붕괴 리스크:** RL은 가장 쉬운 보상 경로를 활용하여 최소 검색 전용 하네스로 다시 붕괴하려는 경향이 있다.

4. **서브에이전트 역할 한계:** Harness-1은 다운스트림 답변 모델을 위한 큐레이션 문서를 반환하는 역할이며, 질문에 직접 답변하지 않는다는 점에서 단독 응용의 한계가 있다.

5. **컨텍스트 비용 문제:** 외부화된 증거 로그가 턴마다 수천 토큰을 추가할 수 있어 장기 에피소드에서 비용이 급증할 수 있다는 커뮤니티 우려가 제기되었다.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능은 이 논문의 가장 주목할 만한 결과 중 하나이다.

Harness-1의 성능 향상은 특히 훈련 외 전이(held-out transfer) 벤치마크에서 더욱 강하게 나타나며, 이는 명시적 검색 상태에 대한 강화학습이 훈련 도메인을 넘어서는 검색 행동의 일반화를 만들어낼 수 있음을 시사한다.

가장 흥미로운 결과는 전이(transfer)에서: 성능 향상이 소스 패밀리 벤치마크보다 훈련 외 전이 벤치마크에서 실질적으로 더 크게 나타난다.

### 일반화 성능 향상의 메커니즘적 해석

Harness-1이 드러내는 핵심 통찰은 에이전틱 RL 훈련이 실제로는 추론 문제가 아닌 것들 — 후보 목록 추적, 검증 로그 관리 등 — 에 대해 모델에 조용히 페널티를 부여해왔다는 것이다. 이 오버헤드를 외부화함으로써 20B 모델은 진정으로 의미론적 판단이 필요한 결정에 학습된 용량을 집중할 수 있다.

이를 일반화 관점에서 분석하면:

$$\underbrace{\pi_{\text{기존}}}_{\text{정책}} = \underbrace{f_{\text{semantic}}}_{\text{의미 결정}} + \underbrace{f_{\text{bookkeeping}}}_{\text{상태 관리}}$$

$$\underbrace{\pi_{\text{Harness-1}}}_{\text{정책}} = \underbrace{f_{\text{semantic}}}_{\text{의미 결정}} \quad \leftarrow \underbrace{\mathcal{H}}_{\text{Harness가 상태 관리 담당}}$$

정책이 순수 의미론적 검색 결정($f_{\text{semantic}}$)에만 집중할 때, 더 범용적이고 도메인 독립적인 검색 전략을 학습할 수 있다.

Harness-1은 웹, 금융, 특허, 멀티홉 QA를 아우르는 8개 벤치마크에서 평가되었으며, 주요 메트릭은 최종 세트에서의 관련 문서 커버리지인 큐레이션 재현율이다.

이처럼 다양한 도메인 전이에서의 성능 유지는 **하네스가 행동적 사전 정보(behavioral prior)를 담을 수 있다**는 점에서도 설명된다: 핵심은 "적은 데이터로 충분하다"가 아니라, 행동적 사전 정보의 상당 부분이 하네스 안에 담길 수 있다는 것이다.

---

## 4. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 4.1 연구에 미치는 영향

**① 에이전트 설계 패러다임 전환**

이 아키텍처적 전환은 특히 쿼리 깊이에 따라 상태 복잡성이 증가하는 RAG(Retrieval-Augmented Generation) 파이프라인에서 생산 검색 시스템이 모델 용량과 환경 인프라 사이의 균형을 맞추는 방식을 재편할 수 있다.

**② 소규모 모델의 경쟁력 재조명**

Harness-1은 어려운 검색 작업에서 훨씬 큰 프런티어 모델 검색기들과 동등하거나 능가하는 20B 오픈 검색 에이전트이다. 이는 모델 크기보다 아키텍처 설계가 더 중요할 수 있다는 함의를 갖는다.

**③ RL 훈련 효율 개선의 새로운 방향**

이는 RL에게 추가 전용 트랜스크립트에서 사무적 기록을 재발견하도록 요청하는 대신, 검색 행동 개선을 위한 안정적인 인터페이스를 제공한다.

**④ 하네스 공동 최적화 연구의 촉진**

관련 연구들(Meta-Harness, Agentic Harness Engineering 등)과의 연계에서, 하네스 자체를 학습/최적화 대상으로 삼는 연구 방향이 급속히 부상하고 있다.

---

### 4.2 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|---|---|
| **하네스 설계의 최적화** | 메모리 레이아웃, 액션 공간, 렌더링 방식이 모두 성능에 영향. 하네스 자체가 하이퍼파라미터화될 필요 |
| **컨텍스트 비용 vs. 성능 트레이드오프** | 하네스의 외부 메모리가 턴당 컨텍스트 토큰을 대폭 증가시킬 수 있어 비용-성능 최적화가 필요 |
| **보상 함수 설계** | 도구 다양성 보너스와 같이 붕괴를 방지하는 인센티브 설계가 핵심 |
| **단일 도메인 RL 훈련의 일반화** | SEC 쿼리로만 학습된 RL이 다양한 도메인으로 전이하는 메커니즘을 더 깊이 이해할 필요 |
| **하네스-모델 결합 의존성** | 하네스와 모델이 공동으로 최적화되므로, 다른 베이스 모델에 하네스를 재사용할 때의 적응 문제 고려 |
| **멀티에이전트 확장** | 검색 서브에이전트와 답변 에이전트 간의 인터페이스를 더 정밀하게 설계하는 연구 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 아이디어 | Harness-1과의 관계 |
|---|---|---|---|
| **ReAct** (Yao et al.) | 2022 | 추론(Reasoning) + 행동(Acting) 통합 | 기반 패러다임. 하지만 상태 관리를 모델에 위임 — Harness-1이 이 문제를 해결 |
| **WebGPT** (OpenAI) | 2021 | 웹 검색 에이전트 + RL | 검색 에이전트 RL의 초기 시도. 상태 외부화 없음 |
| **IRCOT / Iterative Retrieval** | 2022~2023 | 멀티홉 QA를 위한 반복 검색 | 반복 검색의 필요성 확인. 상태 관리 문제 미해결 |
| **Escaping the Context Bottleneck** | 2026 | RL을 통한 능동적 컨텍스트 큐레이션 | Harness-1과 유사한 문제 인식. 컨텍스트 병목 해결에 초점 |
| **Meta-Harness** (Lee et al.) | 2026 | 하네스 코드 자체를 엔드투엔드 최적화 | Harness-1의 하네스를 고정 설계로 보는 반면, Meta-Harness는 이를 동적으로 최적화 |
| **Agentic Harness Engineering (AHE)** | 2026 | 관찰 가능성 기반 하네스 자동 진화 | 동결된 하네스가 재진화 없이 전이됨. SWE-bench-verified에서 더 적은 토큰으로 성능 달성, 이는 진화된 구성 요소가 벤치마크 특화 튜닝이 아닌 일반적 엔지니어링 경험을 인코딩함을 시사 |
| **CISPO** | 2025~2026 | On-policy RL with importance sampling | Harness-1의 RL 알고리즘으로 직접 채택됨 |

### 주요 비교 포인트

이 구분은 강화학습에서 특히 중요해진다. 최근 연구들은 LLM이 RL을 통해 검색 엔진 및 검색 시스템과 상호작용하도록 훈련될 수 있음을 보여주었으며, 쿼리 생성, 멀티턴 검색, 다운스트림 검색 유용성을 개선했다.

Harness-1이 기존 연구와 차별화되는 핵심은 **"모델이 아닌 환경이 상태를 유지한다"**는 설계 철학이다. 대부분의 기존 연구(ReAct, WebGPT, Iterative Retrieval 등)는 모델의 컨텍스트 윈도우 내에 상태 정보를 축적하는 방식을 사용했으나, Harness-1은 이를 환경으로 분리함으로써 RL이 사무적 작업(bookkeeping)을 환경의 상태적 하네스에 위임하고, 정책은 순수한 의미론적 검색 결정에만 집중하도록 한다.

---

## 📚 참고 자료 및 출처

1. **Harness-1 원문 논문 (arXiv):** Pengcheng Jiang et al., "Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses," arXiv:2606.02373, 2026. https://arxiv.org/abs/2606.02373
2. **HuggingFace Paper Page:** https://huggingface.co/papers/2606.02373
3. **MarkTechPost 분석 기사:** "Meet Harness-1: A 20B Retrieval Subagent Trained With Reinforcement Learning Inside a Stateful Search Harness on gpt-oss-20b," MarkTechPost, June 6, 2026. https://www.marktechpost.com/2026/06/06/
4. **The Model Wire:** "Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses," themodelwire.com
5. **Digg AI:** "Creator Patrick Jiang launches Harness-1," digg.com
6. **관련 논문 - Meta-Harness:** Lee et al., "Meta-Harness: End-to-End Optimization of Model Harnesses," arXiv:2603.28052, 2026.
7. **관련 논문 - Agentic Harness Engineering:** Shichun Liu et al., "Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses," arXiv:2604.25850, 2026.

> ⚠️ **정확도 관련 고지:** 본 답변은 공개된 arXiv 초록, HuggingFace 논문 페이지, MarkTechPost 분석 기사, 저자 공개 발언 등을 기반으로 작성되었습니다. 보상 함수의 **정확한 수학적 형식** (예: $R_\text{curated}$의 구체적 정의)은 논문 PDF 전문에서 확인이 필요하며, 상기 수식은 논문에서 언급된 구성 요소를 기반으로 재구성한 것임을 명시합니다. 정확한 수식은 arXiv 원문 PDF를 직접 참조하시기 바랍니다.
